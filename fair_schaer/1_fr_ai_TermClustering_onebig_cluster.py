# coding: utf-8

# OS machine interaction
from pathlib import Path
import os

# Data Handling & Visualization
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
from sqlalchemy import create_engine

# Text Processing & Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# Text Processing
import nltk
from unidecode import unidecode
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Progress monitoring
from tqdm.auto import tqdm

# Multiprocessing
import concurrent.futures
from functools import partial
import time

# API interface
import openai
from openai import OpenAI
import ast
import json

# Rate limiting
import backoff
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = "enter here"  # Replace with your API key
openai.api_key = os.environ["OPENAI_API_KEY"]

def sql_query(query, engine):
    """Execute SQL query and return results as DataFrame"""
    return pd.read_sql(query, con=engine)

def preprocess_text(text, stemmer, stop_words):
    """
    Preprocess text: lowercase, remove accents, tokenize, remove stopwords, stem
    Also filters out single digits, single characters, and common French terms
    """
    if not isinstance(text, str) or not text:
        return []
    
    # Additional terms to filter out
    single_digits = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    
    # Handle apostrophes in French
    text = text.replace("l'", "l ")
    text = text.replace("d'", "d ")
    
    # Normalize and tokenize
    words_raw = [unidecode(word.lower()) for word in text.split()]
    
    # Filter out single characters, digits, and stopwords, then stem
    words_processed = []
    for word in words_raw:
        if (len(word) > 1 and  # Remove single characters
            word not in stop_words and  # Remove stopwords
            word not in single_digits):  # Remove single digits
            stemmed_word = stemmer.stem(word)
            if len(stemmed_word) > 1:  # Ensure stemmed words are also longer than 1 character
                words_processed.append(stemmed_word)
    
    return words_processed

def vectorized_term_frequency(texts, stemmer, stop_words):
    """
    Vectorized approach to calculate term frequency
    """
    print("Preprocessing texts and calculating term frequency...")
    
    # Process all texts in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a partial function with fixed stemmer and stop_words
        partial_preprocess = partial(preprocess_text, stemmer=stemmer, stop_words=stop_words)
        # Process all texts with progress bar
        tokens_list = list(tqdm(
            executor.map(partial_preprocess, texts),
            total=len(texts),
            desc="Tokenizing texts"
        ))
    
    # Flatten list of tokens and count frequencies
    all_tokens = [token for tokens in tokens_list for token in tokens]
    
    # Use Counter for efficient frequency counting
    term_counts = Counter(all_tokens)
    
    # Convert to DataFrame for easier manipulation
    df_terms = pd.DataFrame(
        {'term': list(term_counts.keys()), 'count': list(term_counts.values())}
    ).sort_values(by='count', ascending=False)
    
    return df_terms, tokens_list

def truncate_text(text, max_tokens=8000):
    """Truncate text to stay within token limits"""
    if not text:
        return text
        
    # Approximate token count (roughly 4 chars per token for most languages)
    # Using a conservative estimate to avoid hitting limits
    if len(text) > max_tokens * 4:
        return text[:max_tokens * 4]
    return text

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: print(f"Retrying after error: {retry_state.outcome.exception()}. Waiting {retry_state.next_action.sleep} seconds...")
)
def get_embedding_with_retry(text, client, model="text-embedding-3-small"):
    """Get embedding with retry logic for API errors"""
    if not isinstance(text, str) or not text:
        return [0] * 1536  # Return zero vector (1536 dimensions for text-embedding-3-small)
    
    # Clean and truncate text to avoid token limit errors
    text = text.replace("\n", " ").strip()
    text = truncate_text(text)
    
    try:
        result = client.embeddings.create(input=[text], model=model)
        return result.data[0].embedding
    except Exception as e:
        # If we hit token limits despite truncation, log and return zero vector
        if "maximum context length" in str(e):
            print(f"Text too long even after truncation: {text[:100]}... ({len(text)} chars)")
            return [0] * 1536
        raise  # Re-raise other exceptions for retry mechanism

def batch_get_embeddings(texts, batch_size=25):
    """
    Get embeddings for a list of texts in batches with parallel processing
    Shows only overall progress, not individual batch progress
    Uses smaller batch size to avoid rate limits and reduce errors
    """
    client = OpenAI()
    
    # Using a smaller batch size to avoid overwhelming the API
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    all_embeddings = []
    
    print(f"Getting embeddings for {len(texts)} texts in {len(batches)} batches...")
    
    # Process each batch with overall progress bar only
    progress_bar = tqdm(total=len(texts), desc="Processing embeddings")
    
    for i, batch in enumerate(batches):
        # Use ThreadPoolExecutor with limited concurrency to avoid rate limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all texts in the batch
            future_to_text = {
                executor.submit(get_embedding_with_retry, text, client): text 
                for text in batch
            }
            
            # Process completed futures without inner progress bar
            batch_embeddings = []
            for future in concurrent.futures.as_completed(future_to_text):
                text = future_to_text[future]
                try:
                    embedding = future.result()
                    batch_embeddings.append(embedding)
                except Exception as e:
                    print(f"Error getting embedding for text: {text[:50]}..., Error: {e}")
                    # Append zero vector for failed embeddings
                    batch_embeddings.append([0] * 1536)  # 1536 dimensions for text-embedding-3-small
                
                # Update the main progress bar
                progress_bar.update(1)
        
        all_embeddings.extend(batch_embeddings)
        
        # Add a delay between batches to avoid rate limits (longer delay)
        if i < len(batches) - 1:
            time.sleep(2)
    
    progress_bar.close()
    return all_embeddings

def compute_silhouette_for_k(args):
    """
    Helper function to compute silhouette score for a specific k value
    Used for parallel processing
    """
    k, embeddings_array = args
    try:
        kmeans = KMeans(
            n_clusters=k, 
            random_state=42, 
            n_init='auto',
            verbose=0
        ).fit(embeddings_array)
        
        kmeans_labels = kmeans.labels_
        score = silhouette_score(embeddings_array, kmeans_labels, metric='cosine')
        return {'k': k, 'silhouette_score': score}
    except Exception as e:
        print(f"Error computing clusters for k={k}: {e}")
        return {'k': k, 'silhouette_score': -1}  # Return -1 for failed attempts

def find_optimal_clusters(embeddings_array, max_k=100, min_k=2):
    """
    Find optimal number of clusters based on silhouette score
    Parallelized version using ProcessPoolExecutor
    """
    print(f"Finding optimal clusters from {min_k} to {max_k} in parallel...")
    
    # Create parameter list for all k values
    k_values = list(range(min_k, max_k + 1))
    args_list = [(k, embeddings_array) for k in k_values]
    
    silhouette_scores = []
    
    # Create a custom progress bar
    pbar = tqdm(total=len(k_values), desc="Testing cluster counts")
    
    # Use ProcessPoolExecutor to parallelize computation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all jobs
        future_to_k = {executor.submit(compute_silhouette_for_k, args): args[0] for args in args_list}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_k):
            k = future_to_k[future]
            try:
                result = future.result()
                if result['silhouette_score'] >= 0:  # Only add valid results
                    silhouette_scores.append(result)
            except Exception as e:
                print(f"Error processing k={k}: {e}")
            finally:
                pbar.update(1)
    
    pbar.close()
    
    # Sort by k for consistent output
    silhouette_scores.sort(key=lambda x: x['k'])
    
    return silhouette_scores

def get_cluster(q_topics="", engine=None):
    """
    Returns a DataFrame with Clusters and saves them.
    Enhanced version with vectorization and parallel processing.
    """
    print("Processing Topics for all subcollections as one.")
    
    # Get topics from database
    topics = sql_query(q_topics, engine)
    print(f"Retrieved {len(topics)} topics from database.")
    
    # Download NLTK resources if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    # Create stemmer and enhanced stopwords set
    french_stopwords = set(stopwords.words('french'))
    
    # Add additional French stopwords (articles, prepositions, etc.)
    additional_stopwords = {
        'a', 'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'en',
        'et', 'il', 'ils', 'je', 'j', 'la', 'le', 'les', 'leur', 'lui', 'ma',
        'mais', 'me', 'même', 'mes', 'moi', 'mon', 'ni', 'notre', 'nous', 'on',
        'ou', 'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'si', 'son',
        'sur', 'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre',
        'vous', 'c', 'd', 'j', 'l', 'm', 'n', 's', 't', 'y',
        # Numbers written as words
        'zero', 'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept', 'huit', 'neuf', 'dix'
    }
    
    # Combine standard and additional stopwords
    stop_words = french_stopwords.union(additional_stopwords)
    fr_sbst = SnowballStemmer("french")
    
    # Calculate term frequency with vectorized approach
    print("Creating Term Frequency for all subcollections...")
    df_terms_stem, tokens_list = vectorized_term_frequency(topics["text_fr"].tolist(), fr_sbst, stop_words)
    
    # Print top 50 terms
    print("\n50 most frequent words (Stem + StopWord):\n")
    print(df_terms_stem.head(50), "\n")
    
    # Visualize term frequency
    hist_tf = px.histogram(
        df_terms_stem["count"], 
        title="Term Frequency Distribution",
        labels={"value": "Frequency", "count": "Number of Terms"}
    )
    hist_tf.write_html("term_frequency_distribution.html")
    
    # Create category based on most frequent term
    print("Creating category via Term Frequency for all subcollections...")
    
    # Create a dictionary for quick term frequency lookups
    term_freq_dict = dict(zip(df_terms_stem['term'], df_terms_stem['count']))
    
    def query_category(tokens):
        """Find the most frequent term in a list of tokens"""
        if not tokens or len(tokens) == 0:
            return None
        
        # Get frequencies for all tokens, default to 0 if not found
        freqs = [term_freq_dict.get(term, 0) for term in tokens]
        
        # Return the term with highest frequency
        if freqs:
            max_idx = np.argmax(freqs)
            return tokens[max_idx] if max_idx < len(tokens) else None
        
        return None
    
    # Add token lists and categories to topics DataFrame
    topics["token_stem"] = tokens_list
    topics["category"] = topics["token_stem"].apply(query_category)
    
    # Get OpenAI embeddings
    print("Getting OpenAI Embeddings...")
    f_embedding_csv = "topic_embeddings_all_subcollections.csv"
    f_embedding_npy = "topic_embeddings_all_subcollections.npy"
    my_file_csv = Path(f_embedding_csv)
    my_file_npy = Path(f_embedding_npy)
    
    if not my_file_csv.is_file():
        # If no embeddings exist yet, generate them
        print("Generating new embeddings...")
        embeddings = batch_get_embeddings(topics["text_fr"].tolist())
        topics["embedding"] = embeddings
        
        # Save to CSV (original format)
        topics.to_csv(f_embedding_csv, index=False)
        print(f"Saved embeddings to {f_embedding_csv}")
        
        # Also save as NumPy array for faster future loading
        embeddings_array = np.array(embeddings)
        np.save(f_embedding_npy, embeddings_array)
        print(f"Also saved embeddings as NumPy array for faster loading")
    else:
        # Embeddings already exist
        if my_file_npy.is_file():
            # If we already have the fast NumPy version, use it
            print(f"Loading pre-existing embeddings from faster NumPy format...")
            embeddings_array = np.load(f_embedding_npy)
            
            # Load topic data without the embedding column to save memory
            topics = pd.read_csv(f_embedding_csv, usecols=lambda x: x != 'embedding')
            print(f"Loaded {len(topics)} topics and {embeddings_array.shape[0]} embeddings")
        else:
            # We only have the CSV with string embeddings
            print(f"Loading pre-existing embeddings from CSV...")
            print("This will be slow due to parsing. Converting to faster format for next time.")
            
            # Load all topic data
            topics = pd.read_csv(f_embedding_csv)
            
            # Convert string embeddings to actual arrays in batches
            print("Converting string embeddings to arrays...")
            chunk_size = 1000
            all_embeddings = []
            
            with tqdm(total=len(topics), desc="Parsing embeddings") as pbar:
                for i in range(0, len(topics), chunk_size):
                    # Process in chunks to reduce memory pressure
                    chunk = topics["embedding"][i:i+chunk_size]
                    
                    # Parse embeddings in parallel
                    with concurrent.futures.ProcessPoolExecutor() as executor:
                        chunk_embeddings = list(executor.map(ast.literal_eval, chunk))
                    
                    all_embeddings.extend(chunk_embeddings)
                    pbar.update(len(chunk))
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings)
            
            # Save as NumPy file for faster loading next time
            np.save(f_embedding_npy, embeddings_array)
            print(f"Saved parsed embeddings as NumPy array for faster loading next time")
            
            # Remove embedding column to save memory
            topics = topics.drop(columns=["embedding"])
    
    # Print info about the embeddings
    print(f"Embeddings array shape: {embeddings_array.shape}")
    
    # UMAP dimensionality reduction (replacing t-SNE)
    print("Performing UMAP dimensionality reduction...")
    
    # Install UMAP if not already installed
    try:
        import umap
    except ImportError:
        print("Installing UMAP package...")
        import subprocess
        subprocess.check_call(["pip", "install", "umap-learn"])
        import umap
    
    # Create and fit UMAP model
    umap_model = umap.UMAP(
        n_components=2,          # 2D visualization
        n_neighbors=15,          # Balance between local and global structure
        min_dist=0.1,            # Minimum distance between points
        metric='cosine',         # Same metric as used for clustering
        random_state=42,         # For reproducibility
        low_memory=True,         # For large datasets
        verbose=True             # Show progress
    )
    
    reduced_embeddings = umap_model.fit_transform(embeddings_array)
    
    # Visualize clusters by term frequency category
    fig = px.scatter(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        color=topics.category.values,
        hover_name=[str(query) for query in topics["text_fr"]],
        title='UMAP embeddings by term frequency category',
        width=800,
        height=600,
        color_discrete_sequence=plotly.colors.qualitative.Alphabet_r
    )
    
    fig.update_layout(
        xaxis_title='First component',
        yaxis_title='Second component'
    )
    
    fig.write_html("tfcategory_TermCluster_UMAP.html")
    
    # Find optimal number of clusters
    silhouette_scores = find_optimal_clusters(embeddings_array, max_k=99, min_k=2)
    
    # Visualize silhouette scores
    fig = px.line(
        pd.DataFrame(silhouette_scores).set_index('k'),
        title='<b>Silhouette scores for K-means clustering</b>',
        labels={'value': 'Silhouette score'},
        color_discrete_sequence=plotly.colors.qualitative.Alphabet
    )
    fig.update_layout(showlegend=False)
    fig.write_html("Silhouette_TermCluster_onebig_Cluster.html")
    
    # Find the best score using max() with a key function
    best_result = max(silhouette_scores, key=lambda x: x['silhouette_score'])
    
    # Get the optimal k automatically
    auto_k = best_result['k']
    best_score = best_result['silhouette_score']
    
    # Allow manual override
    print(f"\nAutomatic best k: {auto_k} (silhouette score: {best_score:.4f})")
    k = input(f"Enter wanted k-Cluster for clustering topics (press Enter for auto k={auto_k}):\n")
    
    # Use auto_k if no input or non-numeric input
    try:
        k = int(k) if k.strip() else auto_k
    except ValueError:
        k = auto_k
        print(f"Invalid input. Using automatic k={auto_k}")
    
    print(f"Using k-Cluster: {k}\n")
    
    # Perform KMeans clustering
    print(f"Performing K-means clustering with k={k}...")
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init='auto'
    ).fit(embeddings_array)
    
    kmeans_labels = kmeans.labels_
    
    # Visualize clusters with K-means labels using UMAP
    fig = px.scatter(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        color=list(map(lambda x: f'cluster {x}', kmeans_labels)),
        hover_name=[str(query) for query in topics["text_fr"]],
        title='UMAP embeddings with K-means clusters',
        width=800,
        height=600,
        color_discrete_sequence=plotly.colors.qualitative.Alphabet_r
    )
    
    fig.update_layout(
        xaxis_title='First component',
        yaxis_title='Second component'
    )
    
    fig.write_html("umap_TermCluster_Cluster.html")
    
    # Add cluster labels to topics DataFrame
    topics["cluster"] = kmeans_labels
    
    # Save final results
    output_file = "topics_cluster_all_subcollections.csv"
    topics.to_csv(output_file, index=False)
    print(f"Saved clustered topics to {output_file}")
    
    return topics


if __name__ == "__main__":
    # Database connection parameters
    DATABASE = "longeval-web"
    USER = "dis18"
    HOST = "db"
    PORT = "5432"
    PASSWORD = "dis182425"
    
    # Create database engine
    engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
    
    # Test connection
    try:
        df = pd.read_sql('select * from "Topic" limit 1', con=engine)
        print("Database connection successful")
    except Exception as e:
        print(f"Database connection error: {e}")
        exit(1)
    
    # SQL query for topics
    q_topics = """
                select *
                from "Topic"
                """
    
    # Execute clustering
    start_time = time.time()
    topics_with_clusters = get_cluster(q_topics=q_topics, engine=engine)
    end_time = time.time()
    
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Found {len(topics_with_clusters)} topics in {len(topics_with_clusters['cluster'].unique())} clusters")