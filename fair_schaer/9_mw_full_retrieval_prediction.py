# Import Python Functionality
import os
import time
import yaml
from tqdm import tqdm
from typing import List
from collections import Counter

# Import Memory Efficiency
import gc

# Import Parallel Processing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Text Processing
import re
import nltk
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
french_stopwords = set(stopwords.words('french'))

from unidecode import unidecode

# Import Data Processing
import pandas as pd
import numpy as np

# Import Data Interfaces
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Deep Learning
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

# Import Search Engine
import pyterrier as pt
if not pt.java.started():
    pt.java.init()
        
# SQL Pooling
def create_engine_with_pool():
    """
    Creates an engine with proper connection pooling configuration.
    """
    
    return create_engine(
        f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}",
        poolclass=QueuePool,
        pool_size=5,  # Number of permanent connections
        max_overflow=10,  # Number of additional connections that can be created
        pool_timeout=30,  # Timeout waiting for a connection (seconds)
        pool_recycle=1800,  # Recycle connections after 30 minutes
        pool_pre_ping=True  # Verify connection validity before using
    )

#+==============================================================================+
#|                     Longeval Search Results and Evaluation                    |
#+==============================================================================+

def load_index(dataset: str="longeval-web", language: str="fr", sub_collection:str =None)-> pt.IndexFactory:
    """Reads the sub_collection into memory and returns index.

    Args:
        dataset (str, optional): _description_. Defaults to "longeval-web".
        language (str, optional): _description_. Defaults to "fr".
        sub_collection (str, optional): _description_. Defaults to None.
    """
    global index 

    
    with open(BASE_PATH + "/metadata.yml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    # load index
    index_path = os.path.join(".", BASE_PATH, f"index/{dataset}-{language}-{sub_collection}-pyterrier")
    index = pt.IndexFactory.of(index_path)
    
def load_topics(language: str="fr", sub_collection:str = None, limit: int=None)-> pd.DataFrame:
    """Reads Topics via SQL Query and returns topics Data Frame

    Args:
        language (str, optional): _description_. Defaults to "fr".
        sub_collection (str, optional): _description_. Defaults to None.

    Returns:
        pd.DataFrame: _description_
    """
    global topics 

    # Topics in SUB_COLLECTION Filtered for coocurrence in QREL, TOPICS
    q_topics= f"""
                select distinct b.queryid qid, b.text_{language} query 
                from "Qrel" a 
                --Remark: Using all Querid's inside the full run. Only requirement is, that they are distinct and have been judged once in a sub_collection.
                join  (
                        select  *
                       from    "Topic"
                    ) b
                    on      a.queryid = b.queryid
                --join (
                --        select distinct docid
                --        from   "Document"
                --        where  sub_collection = '{sub_collection}'
                --    )c
                --    on ('doc' || a.docid) = c.docid
                --where a.sub_collection = '{sub_collection}'
                group by b.queryid, b.text_{language} 
                """
    
    # All Topics NO FILTER
    q_topics_all = f"""
                select distinct a.queryid qid, a.text_{language} query 
                from "Topic" a 
                group by a.queryid, a.text_{language} 
                """

    # Topics filtered for SUB_COLLECTION (after new SQL DATA was uploaded to database)
    q_topics_sub_collection = f"""
                select distinct a.queryid qid, a.text_{language} query 
                from "Topic" a 
                where sub_collection = '{sub_collection}'
                group by a.queryid, a.text_{language} 
                """

    q_topics_sub_collection_complex = f"""
                select distinct b.queryid qid, b.text_{language} query 
                from "Qrel" a 
                --Remark: Using all Querid's inside the full run. Only requirement is, that they are distinct and have been judged once in a sub_collection.
                join  (
                        select  *
                       from    "Topic"
                    ) b
                    on      a.queryid = b.queryid
                join (
                        select distinct docid
                        from   "Document"
                        where  sub_collection = '{sub_collection}'
                    )c
                    on ('doc' || a.docid) = c.docid
                where a.sub_collection = '{sub_collection}'
                group by b.queryid, b.text_{language} 
                """

    f_all_topics = False
    if f_all_topics:
        q_topics = q_topics_all
    
    # Read topics from SQL-Query 
    topics = sql_query(q_topics_sub_collection)
    print("Topics: ",len(topics))
    
    # Transform to str and remove non alphanumerice characters
    topics["qid"] = topics["qid"].astype(str)
    topics["query"] = topics["query"].str.replace("'", "")
    topics["query"] = topics["query"].str.replace("*", "")
    topics["query"] = topics["query"].str.replace("/", "")
    topics["query"] = topics["query"].str.replace(":", "")
    topics["query"] = topics["query"].str.replace("?", "")
    topics["query"] = topics["query"].str.replace(")", "")
    topics["query"] = topics["query"].str.replace("(", "")
    topics["query"] = topics["query"].str.replace("+", "")
    
    # Remove QueryId's determined as SPAM
    spam = ["59769", "6060", "75200", "74351", "67599", "74238", "74207", "75100", "58130"]
    topics = topics[~topics["qid"].isin(spam)]
    if limit is not None:
        topics = topics.sample(n=limit, random_state=1)
    
    
#+==============================================================================+
#|                             Create Prediction Set                            |
#+==============================================================================+
#+----------------------------------------------------------------------+
#|                     Multi-Hot Encoding Documents                      |
#+----------------------------------------------------------------------+
# +---------------------------------------------------+
#|             Inner Parallel Processing               |
# +---------------------------------------------------+
def stem_text (text, top_words_list):
    """
    Stems text and returns only terms included in top_words_list.
    """
    fr_sbst = SnowballStemmer("french")
    french_stopwords = set(stopwords.words('french'))
    
    # Add additional French stopwords (articles, prepositions, etc.)
    additional_stopwords = {
        'a', 'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'en',
        'et', 'il', 'ils', 'je', 'j', 'la', 'le', 'les', 'leur', 'lui', 'ma',
        'mais', 'me', 'même', 'mes', 'moi', 'mon', 'ni', 'notre', 'nous', 'on',
        'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'si', 'son',
        'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre',
        'vous',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        # Numbers written as words
        'zero', 'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf', 'dix',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'
    }
    
    # Combine standard and additional stopwords
    stop_words = french_stopwords.union(additional_stopwords)
    # Start
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    words_document = [unidecode(word.lower()) for word in list(text.split())]
    words_document_stem = [fr_sbst.stem(word) for word in words_document if not word.lower() in stop_words]
    return [word for word in words_document_stem if word in top_words_list]

def  create_words_index(df_terms_stem, n_words):
    """
    Creates translation table for term <> index.
    """
    df_words_index = df_terms_stem.iloc[:n_words, :1].copy().reset_index()
    df_words_index["index"] = df_words_index.index.tolist()
    df_words_index.to_csv("words_index_translationtable.csv", index=False)
    return df_words_index

def get_term_index(word_list, term_idx_table):
    """
    Searches for the corresponding index for each term.
    """
    df_index = pd.DataFrame(word_list, columns=["term"])
    df_index = df_index.merge(term_idx_table, how="left", left_on="term", right_on="term")
    return  df_index["index"].to_list()

def documents_topwords_filter(df_batch, top_words_list):
    """
    Splits text of each document and stems the content.
    Filters stemmed terms, to only include top_words_list entries.
    Returns DataFrame with Filtered terms for each document.
    """
    #df_batch_filtered = pd.DataFrame(columns=["docid","term_list_stemmed"])
    #df_batch_filtered["docid"] = df_batch["docid"].copy()
    #df_batch_filtered["term_list_stemmed"] = df_batch["text_fr"].apply(lambda x: stem_text(x, top_words_list))
    df_batch["term_list_stemmed"] = df_batch["text_fr"].apply(lambda x: stem_text(x, top_words_list))
    
    return df_batch

def process_batch_MHE(batch_id: int, batch, top_words_list, df_words_index):
    """
    Processes Single Batch from Parallel Batch Processing.
    """
    batch_list = "".join(f"'doc{batch[i]}'," if i+1 <  len(batch) else f"'{batch[i]}'" for i in range(len(batch))) # Here because database changed to docid again ... -.-
    #print(n_begin, n_end)
    q_batch = f"""
    select distinct docid, text_fr, sub_collection
    from "Document" a
    where     a.sub_collection = '{SUB_COLLECTION}' -- change here for different sub_collection runs!!!
          and a.docid in ({batch_list})
    order by  a.docid
    """
    engine = create_engine_with_pool()
    try:
        df_batch = pd.read_sql(q_batch, con=engine)
        df_batch = documents_topwords_filter(df_batch, top_words_list)
        df_batch["term_idx"] = df_batch["term_list_stemmed"].apply(lambda x: get_term_index(word_list=x,term_idx_table=df_words_index))
        # print(df_batch) # here
        df_batch = df_batch.loc[:,["docid", "sub_collection", "term_idx"]]
        df_batch["docid"] = df_batch["docid"].str[3:] # Here because database changed to docid again ... -.-
        return batch_id, df_batch
    finally:
        engine.dispose()

# +---------------------------------------------------+
#|             Outer Parallel Processing               |
# +---------------------------------------------------+
# Parallel Processing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import List

def parallel_MultiHotEncoding(docids: list, n_docs, top_words_list, df_words_index):
    """
    Starts Parallel Processing.
    """
    docs_list = docids
    func_slice_batch = lambda begin, end: docs_list[begin:end]
    
    workers = 24  # max sind 32 ich nehm 12 # habe max worker für sql erreicht 12 sind zu viel
    batch_size = 1000

    n_docs = n_docs if n_docs == len(docs_list) else len(docs_list)
    batches = []
    for i in range(0, n_docs, batch_size):
        end_idx = min(i + batch_size, n_docs) # last batch is smaller due to hitting n_docs as maximum
        batches.append((i, func_slice_batch(i, end_idx)))

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all batches and store futures
        future_to_batch = {
        executor.submit(process_batch_MHE, batch_id, batch, top_words_list, df_words_index): batch_id for batch_id, batch in batches
        }

        results = {}

        with tqdm(as_completed(future_to_batch), disable=None, total=len(future_to_batch), desc="Batch Progress") as pbar:
            for i, future in enumerate(pbar):
                batch_id, result = future.result()
                results[batch_id] = result
                pbar.set_description(f"Processing batch {i}")

    #print("Available batch keys:", results.keys())
    #print("Trying to access batches 0 to", len(batches)-1)
    #print(results)
    first_batch = batches[0][0]
    final_results = results[first_batch]
    for i in batches[1:]:
        final_results = pd.concat([final_results, results[i[0]]], ignore_index=True) # i is the batch tuple (batch_id, (func_inner, ... addittional parameters))
        
    #print(f"Processed {n_docs} items in {len(batches)} batches")
    #print(f"Time taken: {time.time() - start_time:.2f} seconds")
    #print(f"Filtered Text for {len(top_words_list)} Top Words of the Corpus.")
    #print(f"First few results: {final_results[:5]}")

    return final_results

def multi_hot_encoding(docids:list, n_docs:int, df_terms_stem, n_words):
    """
    Starts transforming
    """
    # Preparation for Parallel Processing
    #print(f"Processing {n_docs} documents with Top {n_words} words")
    top_words_list = df_terms_stem.iloc[:n_words, 0].to_list()
    df_words_index = create_words_index(df_terms_stem, n_words=n_words)
    
    # Iterative Processing via Parallel Processing
    docs_topwords = parallel_MultiHotEncoding(docids, n_docs, top_words_list, df_words_index) 

    return docs_topwords

def get_predictset(docid_batch: pd.DataFrame=None)->pd.DataFrame:
    """
    Creates a Multi-Hot Encoded dataset for prediction.
    """
    # Create unique docids
    docid_batch = docid_batch.drop_duplicates(subset="docno")
    docids = docid_batch["docno"].tolist()
    n_docs = len(docids)
    
    # Overview over Search Content
    #print(f"Creating predict set for Search Results")
    #print(f"Unique documents found: {len(docids)}")

    subcol_terms_used = "all_subcollections"
    
    # Get Top Words for each subcollection
    #print(f"Use Top Terms from prior SubCollection:\t\t{subcol_terms_used}")
    df_terms_stem = pd.read_csv(f"top_terms_stemmed_{subcol_terms_used}.csv")
    
    # Get seperate MultiHotEncoded and Docs DataFrame from subcollection     
    docs_topwords = multi_hot_encoding(docids, n_docs, df_terms_stem, n_words=10_000)     #number of docs in subset:  n_docs
    
    # Clean Memory and Return results
    gc.collect()
    return docs_topwords

#+==============================================================================+
#|                       Predict Cluster & Relevance                            |
#+==============================================================================+

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences): # i is the n_th review whereas the sequence assigns like a list of fields (like pandas data frame) all the right fields results[i, [3, 5]] = 1
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

def predict(predict_set: pd.DataFrame=None):
    """ 
    Predicts relevance for documents in result set.
    """
    # Load predict set into Multi Hot Encoded Numpy Array
    n_words = 10_000
    x_predict = vectorize_sequences(predict_set["term_idx"], dimension=n_words)
    
    # Predict cluster and relevance
    model = load_model("models/best_bs512_model_callback_best_epoch_all_subcollections.keras")
    cluster_pred, relevance_pred = model.predict(x_predict)

    # Transformation and appending to pd.Dataframe
    cluster_pred = np.argmax(cluster_pred, axis=1)
    predict_set["cluster_pred"] = cluster_pred
    predict_set["rel_pred"] = relevance_pred.flatten()*2

    # Clean Memory and Return results
    gc.collect()
    return predict_set.loc[:,["docid","sub_collection","cluster_pred","rel_pred"]]

#+==============================================================================+
#|                                    Main                                      |
#+==============================================================================+

def main():
    """
    Executes the Pipeline.
    Batch wise process:
    1. Takes n-Topics 
    2. Searches n-Topics and creates result set
    3. Creates One-Hot Encoding from result set
    4. Predicts Relevance from result set
    5. Reranks result set with relevance prediction
    6. Appends to run-file.
    7. Start from 1. until every topic is progressed.
    """
    # Load external Data
    ## longeval
    load_index(dataset=DATASET, language=LANGUAGE, sub_collection=SUB_COLLECTION)
    load_topics(language=LANGUAGE, sub_collection=SUB_COLLECTION, limit=None)
    
    ## created topic cluster
    f_cluster = "topics_cluster_all_subcollections.csv"
    df_cluster = pd.read_csv(f_cluster)
    df_cluster = df_cluster.astype({"queryid":"string"})
    cluster_map = df_cluster.set_index("queryid")["cluster"]
    del df_cluster
    gc.collect()
    
    # Split topics into batchsize
    batch_size = 100
    batches = [topics[i:i+batch_size] for i in range(0, len(topics), batch_size)]
    
    # Initiate BM25 Engine
    BM25 = pt.terrier.Retriever(index, wmodel="BM25", verbose=True)
    print(">>> Loaded index with", index.getCollectionStatistics().getNumberOfDocuments(), "documents.")
    
    # Run batches
    f_first = True
    with tqdm(as_completed(batches), disable=None, total=len(batches), desc="Run Batch Progress") as pbar:
        for i, batch in enumerate(batches):
            # Collect result set
            run = BM25.transform(batch)
            print("Shape BM25 Search: ",run.shape)
            print(run)
            
            # Prepare for One Hot Encoding
            map_qid_docid = run.set_index("qid")["docno"]
            
            # Drop copy run and transform for prediction
            predict_run = run.loc[:,["qid", "docno"]].copy()
            predict_run["real_cluster"] = predict_run["qid"].map(cluster_map)
            docids = predict_run.drop(columns=["qid"])
            docids = predict_run.drop_duplicates()
            print("Shape docids: ",docids.shape)
            
            # Create Predictset
            pred_batch_set = get_predictset(docids)
            print("Shape Pred Set preparation: ",pred_batch_set.shape)

            # Predict Relevance
            pred_batch_set = predict(predict_set=pred_batch_set)
            # Before your drop_duplicates line
            docid_dupes = pred_batch_set['docid'].duplicated().sum()
            print(f"Duplicate docids before: {docid_dupes}")
            
            # Current drop_duplicates that doesn't work
            pred_batch_set.drop_duplicates(subset='docid', inplace=True)
            
            # Check if duplicates remain
            docid_dupes_after = pred_batch_set['docid'].duplicated().sum()
            print(f"Duplicate docids after: {docid_dupes_after}")  # This will likely still show duplicates
            print("Shape Predicted Set: ",pred_batch_set.shape)
            
            # create mappings to assign to run for readjusting
            cluster_pred = pred_batch_set.set_index("docid")["cluster_pred"]
            rel_pred = pred_batch_set.set_index("docid")["rel_pred"]
            
            # Assign mappings to run
            run["rel_pred"]=1
            run["real_cluster"] = run["qid"].map(cluster_map)
            run["pred_cluster"] = run["docno"].map(cluster_pred)
            run["pred_cluster"] = run["pred_cluster"].fillna(-1).astype("int64")
            # Only assign relevance prediction if real and predicted cluster of the qid are the same.
            cluster_cond = (run["pred_cluster"]!=-1) & (run["real_cluster"]==run["pred_cluster"])
            run.loc[cluster_cond, ["rel_pred"]]= run["docno"].map(rel_pred)

            # rerank with pred_rel factor
            run["score_bm25"] = run["score"]
            run["score"] = run["score"] * run["rel_pred"]
            #print(run)
            run.sort_values(by=["qid", "score"], inplace=True, ascending=False)
            run["rank"] = run.groupby("qid").cumcount() # thank you claude.ai <3
            
            # Save File and Clean Memory for next batch
            if f_first:
                run.to_csv(f"run_modelprediction_{SUB_COLLECTION}.csv", index=False)
                f_first = False
            else:
                run.to_csv(f"run_modelprediction_{SUB_COLLECTION}.csv", index=False, mode="a", header=False)

            gc.collect()
            
            # Watch Progress
            pbar.set_description(f"Processing Batch {i}")            
            
    print("DONE!!!")


if __name__ == "__main__":
    # Global variable declaration 
    ## SQL Credentials
    DATABASE = "longeval-web"
    USER = "dis18"
    HOST = "db"
    PORT = "5432"
    PASSWORD = "dis182425"
    ## Data Set
    DATASET = "longeval-web"
    LANGUAGE = "fr"
    # 
    SUB_COLLECTIONS = ["2023-03", "2023-04", "2023-05", "2023-06", "2023-07", "2023-08"]
    # "2023-02",
    ## Base Path
    BASE_PATH = "/home/jovyan/work/datasets/LongEval-Web"

    # Create sql_engine
    engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
    ## Single Query
    sql_query = lambda x: pd.read_sql(x, con=engine)
    
    # Run Script
    for SUB_COLLECTION in SUB_COLLECTIONS:
        print(SUB_COLLECTION)
        main()