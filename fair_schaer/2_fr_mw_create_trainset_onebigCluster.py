#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sqlalchemy import create_engine

DATABASE = "longeval-web"
USER = "dis18"
HOST = "db"
PORT = "5432"
PASSWORD = "dis182425"

engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")

df = pd.read_sql('select * from "Topic" limit 1', con=engine)
sql_query = lambda x: pd.read_sql(x, con=engine)


# In[2]:


def sql_connection():
    """
    Creates an engine the process can use for multi processing.
    Remark: Connection gets lost if each worker connects via the same connection.
    """
    DATABASE = "longeval-web"
    USER = "dis18"
    HOST = "db"
    PORT = "5432"
    PASSWORD = "dis182425"
    
    engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
    
    return lambda x: pd.read_sql(x, con=engine)


# In[3]:


# Source: ClaudeAI (ran into max connection problem)
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
def create_engine_with_pool():
    """
    Creates an engine with proper connection pooling configuration.
    """
    DATABASE = "longeval-web"
    USER = "dis18"
    HOST = "db"
    PORT = "5432"
    PASSWORD = "dis182425"
    
    return create_engine(
        f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}",
        poolclass=QueuePool,
        pool_size=5,  # Number of permanent connections
        max_overflow=10,  # Number of additional connections that can be created
        pool_timeout=30,  # Timeout waiting for a connection (seconds)
        pool_recycle=1800,  # Recycle connections after 30 minutes
        pool_pre_ping=True  # Verify connection validity before using
    )


# In[4]:


# Get sub_collection and count(*) for each
query= """
select sub_collection, count(*)
from "Document"
group by sub_collection
"""
df_subcol_count = sql_query(query)
print(df_subcol_count)


# In[5]:


sub_col_name: str= None
sub_count: int = None 
sub_batch_size = 1000

df_subcol_count.apply(lambda x:print(x["sub_collection"],x["count"],"\n"), axis = 1)


# # 2.Pipeline Term Frequency on Documents

# ## 2.1 Inner Parallel Processing

# In[2]:


import pandas as pd
import numpy as np
import gc # memory efficiency
import re
import nltk
from unidecode import unidecode
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
from collections import Counter

nltk.download('stopwords')
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


# In[7]:


def get_document_terms(df_batch):
    """
    Splits text of each document and stems the content.
    Returns list of stemmed terms.
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
    # Create bag if words
    # re replaces all punctuations
    df_batch["text_fr_cleaned"] = df_batch["text_fr"].apply(lambda x: re.sub(r'[^\w\s]|[\d]', '', x) if x else "")
    words_document = [unidecode(word.lower()) for query in df_batch["text_fr_cleaned"] for word in list(query.split())] # Attention!!! List Comprehension is always from left to right. (I forgot that again...)
    
    # 50 häufigsten Wörter nach Stemming
    words_document_stem = [fr_sbst.stem(word) for word in words_document if not word.lower() in stop_words]

    return words_document_stem


# ## 2.2 Outer Parallel Processing

# In[3]:


def process_batch_BoW(batch_id: int, batch, doc_identifier):
    """
    Processes Single Batch from Parallel Batch Processing.
    """
    batch_list = "".join(f"'{batch[i]}'," if i+1 <  len(batch) else f"'{batch[i]}'" for i in range(len(batch)))
    #print(n_begin, n_end)
    # Don#t need the extended query (as in Multi Hot Encoding) because i only need the docid info where evaluated docs are.
    q_batch = f"""
    select     distinct a.docid, a.text_fr
    from       "Document" a
    where      a.docid in ({batch_list}) 
    group by   a.docid, a.text_fr
    """
    #print(q_batch)
    engine = create_engine_with_pool()
    try:
        df_batch = pd.read_sql(q_batch, con=engine)
        
        gc.collect()
        return batch_id, get_document_terms(df_batch)
    finally:
        engine.dispose()


# In[9]:


# Parallel Processing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import List

def parallel_bagofwords(first_doc, n_docs):
    """
    Starts Parallel Processing.
    """
    q_docs = f"""
    select distinct(a.docid)
    from "Document" a
    join (
          select ('doc'|| b_inner.docid)new_docid , *
          from "Qrel" b_inner
          where queryid in (
                select queryid
                from "Qrel" 
                where     sub_collection in ('2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02')
                      and relevance <> '0' -- so that a bias is created in favour of relevant documents
          )
    ) b
    on        a.docid = b.new_docid
          and a.sub_collection = b.sub_collection
    join (
          select *
          from "Topic"
    ) c
    on b.queryid = c.queryid  
    where     b.sub_collection in ('2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02')
          and b.relevance is not null
          and a.sub_collection is not null
          and b.queryid is not null
    limit {int(n_docs)}
    """
    sql_query = sql_connection()
    docs_list = sql_query(q_docs)["docid"].tolist()
    doc_identifier = first_doc[:5]
    doc_id = first_doc[5:]
    #n_begin = int(doc_id)
    func_slice_batch = lambda begin, end: docs_list[begin:end]

    workers = 24  # max sind 32 ich nehm 12 # habe max worker für sql erreicht 12 sind zu viel
    batch_size = 200

    n_docs = n_docs if n_docs == len(docs_list) else len(docs_list)
    batches = []
    for i in range(0, n_docs, batch_size):
        end_idx = min(i + batch_size, n_docs)
        batches.append((i, func_slice_batch(i, end_idx)))

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all batches and store futures
        future_to_batch = {
        executor.submit(process_batch_BoW, batch_id, batch, doc_identifier): batch_id for batch_id, batch in batches
        }

        results = {}

        with tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Batch Progress") as pbar:
            for i, future in enumerate(pbar):
                batch_id, result = future.result()
                results[batch_id] = result
                pbar.set_description(f"Processing batch {i}")

    #print("Available batch keys:", results.keys())
    #print("Trying to access batches 0 to", len(batches)-1)

    final_results = []
    for i in batches:
        #print(i)
        final_results.extend(results[i[0]])

    print(f"Processed {n_docs} items in {len(batches)} batches")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(f"First few results: {final_results[:5]}")

    gc.collect()
    return final_results


# In[10]:


def extract_numbers(text):
    return int(re.sub(r'\D', '', text))

def top_n_words(n_docs:int):
    """
    Starts transforming
    """
    q_first_doc = f"""
    select *
    from "Document"
    limit 1
    """
    n_docs = int(n_docs)
    first_doc = sql_query(q_first_doc)["docid"].item()

    # Get Words from Documents
    words_document_stem = parallel_bagofwords(first_doc, n_docs)
    # Create unique set
    bag_words_stem = set(sorted(words_document_stem))
    # Get Term Frequency
    df_terms_stem = pd.DataFrame(bag_words_stem, columns=["term"])
    df_terms_stem ["count"] = 0
    # Count Words and match with Data Frame
    word_counts = Counter(words_document_stem)
    df_terms_stem['count'] += df_terms_stem['term'].map(word_counts).fillna(0)
    df_terms_stem = df_terms_stem.sort_values(by=["count"], ascending=False)
    

    # Summary
    print(f"Top {len(df_terms_stem)} Words for all SubCollection:") 
    print("Bag of Words")
    print(df_terms_stem)
    print(f"\nTop Words:")
    print(df_terms_stem["term"].tolist()[:10])    

    return df_terms_stem
    
   #for n_begin in range(doc_start, doc_start+n_docs+1, batch_size):
   #    n_end =  n_begin + batch_size
   #    print(n_begin, n_end)
   #    q_batch = f"""
   #    select *
   #    from "Document"
   #            and docid between 'doc0{n_begin}' and 'doc0{n_end}'
   #    """
   #    
   #    df_batch = sql_query(q_batch)
   #    print(df_batch)
   #    
   #    break


# # 3. MultiHotEncoding Documents

# ## 3.1 Inner Parallel Processing

# In[11]:


def stem_text (text, top_words_list):
    """
    Stems text and returns only terms included in top_words_list.
    """
    fr_sbst = SnowballStemmer("french")
    stop_words = set(stopwords.words('french'))
    # Start
    text = re.sub(r'[^\w\s]|[\d]', '', text)
    words_document = [unidecode(word.lower()) for word in list(text.split())]
    words_document_stem = [fr_sbst.stem(word) for word in words_document if not word.lower() in stop_words]
    return [word for word in words_document_stem if word in top_words_list]


# In[12]:


def  create_words_index(df_terms_stem, n_words):
    """
    Creates translation table for term <> index.
    """
    df_words_index = df_terms_stem.iloc[:n_words, :1].copy().reset_index()
    df_words_index["index"] = df_words_index.index.tolist()
    df_words_index.to_csv("words_index_translationtable.csv", index=False)
    return df_words_index


# In[13]:


def get_term_index(word_list, term_idx_table):
    """
    Searches for the corresponding index for each term.
    """
    df_index = pd.DataFrame(word_list, columns=["term"])
    df_index = df_index.merge(term_idx_table, how="left", left_on="term", right_on="term")
    return df_index["index"].to_list()


# In[14]:


def get_topiccluster(docs_topwords):
    """
    Enriches Data Frame with corresponding Topic Cluster.
    """
    topic_cluster = pd.read_csv(f"topics_cluster_all_subcollections.csv")
    #print(topic_cluster.columns.tolist())
    #print(topic_cluster.head())
    # Merge Cluster to Document Data Frame
    docs_topwords = docs_topwords.merge(topic_cluster[["queryid", "cluster"]], how="left", left_on="queryid", right_on="queryid")
    
    return docs_topwords


# In[15]:


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


# In[16]:


def process_batch_MHE(batch_id: int, batch, doc_identifier, top_words_list, df_words_index):
    """
    Processes Single Batch from Parallel Batch Processing.
    """
    batch_list = "".join(f"'{batch[i]}'," if i+1 <  len(batch) else f"'{batch[i]}'" for i in range(len(batch)))
    #print(n_begin, n_end)
    q_batch = f"""
    select a.docid, a.text_fr, a.sub_collection, c.queryid, cast(AVG(cast(b.relevance as int)) as NUMERIC(5,2)) as relevance
    from "Document" a
    join (
          select ('doc'|| b_inner.docid)new_docid , *
          from "Qrel" b_inner
          where queryid in (
                select queryid
                from "Qrel" 
                where sub_collection in ('2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02')
          )
    ) b
    on        a.docid = b.new_docid
          and a.sub_collection = b.sub_collection
    join (
          select *
          from "Topic"
    ) c
    on b.queryid = c.queryid  
    where     b.sub_collection in ('2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02')
          and b.relevance is not null
          and a.sub_collection is not null
          and b.queryid is not null
          and a.docid in ({batch_list})
    group by a.docid, a.text_fr, a.sub_collection, c.queryid
    """
    engine = create_engine_with_pool()
    try:
        df_batch = pd.read_sql(q_batch, con=engine)
        df_batch = documents_topwords_filter(df_batch, top_words_list)
        df_batch["term_idx"] = df_batch["term_list_stemmed"].apply(lambda x: get_term_index(word_list=x,term_idx_table=df_words_index))

        gc.collect()
        return batch_id, df_batch
    finally:
        engine.dispose()


# ## 3.2 Outer Parallel Processing

# In[17]:


# Parallel Processing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import List

def parallel_MultiHotEncoding(first_doc, n_docs, top_words_list, df_words_index):
    """
    Starts Parallel Processing.
    """
    q_docs = f"""
    select distinct a.docid
    from "Document" a
    join (
          select ('doc'|| b_inner.docid)new_docid , *
          from "Qrel" b_inner
          where queryid in (
                select queryid
                from "Qrel" 
                where sub_collection in ('2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02')
          )
    ) b
    on        a.docid = b.new_docid
          and a.sub_collection = b.sub_collection
    join (
          select *
          from "Topic"
    ) c
    on b.queryid = c.queryid  
    where     b.sub_collection in ('2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02')
          and b.relevance is not null
          and a.sub_collection is not null
          and b.queryid is not null
    --limit {int(n_docs)} -- no limits this time (legacy code still included for limit purpose...)
    """
    
    sql_query = sql_connection()
    docs_list = sql_query(q_docs)["docid"].tolist()
    doc_identifier = first_doc[:5]
    doc_id = first_doc[5:]
    #n_begin = int(doc_id)
    func_slice_batch = lambda begin, end: docs_list[begin:end]
    
    workers = 24  # max sind 32 ich nehm 12 # habe max worker für sql erreicht 12 sind zu viel
    batch_size = 100

    n_docs = n_docs if n_docs == len(docs_list) else len(docs_list)
    batches = []
    for i in range(0, n_docs, batch_size):
        end_idx = min(i + batch_size, n_docs)
        batches.append((i, func_slice_batch(i, end_idx)))

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all batches and store futures
        future_to_batch = {
        executor.submit(process_batch_MHE, batch_id, batch, doc_identifier, top_words_list, df_words_index): batch_id for batch_id, batch in batches
        }

        results = {}

        with tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Batch Progress") as pbar:
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
        #print(i)
        final_results = pd.concat([final_results, results[i[0]]], ignore_index=True) # i is the batch tuple (batch_id, (func_inner, ... addittional parameters))

    print(f"Processed {n_docs} items in {len(batches)} batches")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(f"Filtered Text for {len(top_words_list)} Top Words of the Corpus.")
    print(f"First few results:")

    gc.collect()
    return final_results


# In[18]:


def multi_hot_encoding(n_docs:int, df_terms_stem, n_words):
    """
    Starts transforming
    """
    # Init Variables
    n_docs = int(n_docs)
    q_first_doc = f"""
    select *
    from "Document"
    limit 1
    """
    first_doc = sql_query(q_first_doc)["docid"].item()

    # Preparation for Parallel Processing
    top_words_list = df_terms_stem.iloc[:n_words, 0].to_list()
    df_words_index = create_words_index(df_terms_stem, n_words=n_words)
    
    # Iterative Processing via Parallel Processing
    docs_topwords = parallel_MultiHotEncoding(first_doc, n_docs, top_words_list, df_words_index)
    #print(docs_topwords)  

    # Mass Processing Via Data Frame Vectorization
    docs_topwords = docs_topwords.astype({'queryid': 'int64'})
    docs_topwords = get_topiccluster(docs_topwords)
    print(docs_topwords)
    # MARKER!!!!
    # Need to add the cluster and relevance!
    # Remark: need to see how i pass the relevance from the sql database. its a bit more complicated now.
    # Remark: adjust parallel_bag of words as well.
 

    return docs_topwords


# In[19]:


engine.dispose()
#print(n_docs//10)


# In[20]:


# Get sub_collection and count(*) for each
search_collection = '2023-02'
# !!! has to be changed 
query= f"""
select count(*)
from "Document" a
    join (
          select ('doc'|| b_inner.docid)new_docid , *
          from "Qrel" b_inner
          where queryid in (
                select queryid
                from "Qrel" 
                where sub_collection in ('2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02')
          )
    ) b
    on        a.docid = b.new_docid
          and a.sub_collection = b.sub_collection
    join (
          select *
          from "Topic"
    ) c
    on b.queryid = c.queryid  
    where     b.sub_collection in ('2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02')
          and b.relevance is not null
          and a.sub_collection is not null
          and b.queryid is not null
"""
n_docs = sql_query(query).iloc[0,0].item() # [0,0] <- [first row, first column]
print(n_docs)


# In[21]:


# Start creating Training Set
print(f"Creating training set for all Sub Collections in one.")
print(f"Found documents with relevance score:\t\t{n_docs}")

# Get Top Words for each subcollection
df_terms_stem=top_n_words(n_docs)
df_terms_stem.to_csv(f"top_terms_stemmed_all_subcollections.csv", index=False)
gc.collect()
#df_terms_stem = pd.read_csv(f"top_terms_stemmed_all_subcollections.csv")

# Get seperate MultiHotEncoded and Docs DataFrame from subcollection 
docs_topwords = multi_hot_encoding(n_docs, df_terms_stem, n_words=10_000) #n_docs//10
docs_topwords.to_csv(f"bis2023-02_train_set_documents_top_terms_all_subcollections.csv", index=False)
gc.collect()
    
print("\n!!!DONE!!!")


# In[23]:


docs_topwords_compressed = docs_topwords.loc[:,["docid", "term_idx", "cluster", "relevance"]]
docs_topwords_compressed.drop_duplicates(subset=["docid","cluster", "relevance"],inplace=True)
docs_topwords_compressed.to_csv(f"bis2023-02_train_set_documents_top_terms_all_subcollections_compressed.csv", index=False)


# In[25]:


def save_df_np(df, filename):
    """
    Save entire DataFrame to NumPy format, preserving lists and data types.
    
    Parameters:
        df: The pandas DataFrame to save
        filename: Filename without extension
    """
    # Store column types for reconstruction
    column_types = {}
    
    # Convert DataFrame to a dictionary of columns
    data_dict = {}
    
    # Process each column appropriately
    for col in df.columns:
        # Sample the column to detect type
        sample = df[col].iloc[0] if len(df) > 0 else None
        
        if isinstance(sample, list):
            # It's a list column
            data_dict[col] = np.array(df[col].tolist(), dtype=object)
            column_types[col] = 'list'
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Already numeric, save directly
            data_dict[col] = df[col].values
            column_types[col] = 'numeric'
        elif isinstance(sample, str):
            # It's a string column, try to convert numeric strings
            try:
                # Check if all values can be converted to integers
                df[col].astype(int)
                data_dict[col] = df[col].astype(int).values
                column_types[col] = 'int'
            except (ValueError, TypeError):
                try:
                    # Check if all values can be converted to floats
                    df[col].astype(float)
                    data_dict[col] = df[col].astype(float).values
                    column_types[col] = 'float'
                except (ValueError, TypeError):
                    # Regular string column
                    data_dict[col] = df[col].values
                    column_types[col] = 'string'
        else:
            # Other types, save as is
            data_dict[col] = df[col].values
            column_types[col] = 'other'
    
    # Add column types to the data dictionary
    data_dict['__column_types__'] = np.array([column_types], dtype=object)
    
    # Save everything to a single .npz file
    np.savez_compressed(f"bis2023-02_{filename}.npz", **data_dict)
    print(f"Saved DataFrame to {filename}.npz with type detection")


# In[26]:


save_df_np(docs_topwords_compressed, "train_set_documents_top_terms_all_subcollections")

