# Neural Retrieval System Framework: A Step-by-Step Guide

*Generated Documentation by Gemini2.5 via Perplexity*</br>
*Manually checked, validated and adjusted for consistency according to the intentions from the project group **fair_schaer**.*

This guide walks you through a pipeline for building and evaluating a neural retrieval system, specifically tailored for the LongEval shared task using French language data. The framework is designed to be reproducible and adaptable for future projects.

## Introduction

The core idea behind this neural retrieval system is to enhance traditional search methods (like BM25) by re-ranking documents using predictions from a deep learning model. This model learns to predict document relevance and to which thematic cluster a document belongs, based on its textual content.

This pipeline involves several stages:
1.  **Topic Understanding**: Clustering topics to identify underlying themes.
2.  **Training Data Preparation**: Generating features (like multi-hot encoded term vectors) from documents and associating them with relevance judgments and topic clusters.
3.  **Model Training**: Training a neural network to predict document relevance and cluster.
4.  **Retrieval and Re-ranking**: Using the trained model to re-rank documents retrieved by an initial search.
5.  **Evaluation**: Assessing the performance of the retrieval system.
6.  **Output Formatting**: Converting results into standard evaluation formats.

## Prerequisites

Before you begin, ensure you have the following:
*   **Python Environment**: With standard data science and NLP libraries (pandas, numpy, nltk, scikit-learn, TensorFlow/Keras).
*   **PyTerrier**: For information retrieval tasks.
*   **Database Access**: A PostgreSQL database (e.g., "longeval-web") containing "Topic", "Qrel", and "Document" tables, as specified in the scripts.
*   **Dataset**: The LongEval dataset, including PyTerrier indexes and metadata, stored in a `BASE_PATH` (e.g., `/home/jovyan/work/datasets/LongEval-Web`).
*   **OpenAI API Key**: If you intend to re-generate embeddings using OpenAI models, as done in `1_fr_TermClustering_onebig_cluster.py`.

## Pipeline Overviewv

The framework is implemented through a series of Python scripts and Jupyter notebooks:

1.  `1_fr_TermClustering_onebig_cluster.py`: Processes topics to create thematic clusters.
2.  `2_fr_create_trainset_onebigCluster.py`: Generates a training set by processing documents, calculating term frequencies, and creating multi-hot encoded representations.
3.  `3_create_model_hyperparameter_ai-support.py`: Trains a neural network model to predict document relevance and cluster assignment.
4.  `4a_full_retrieval_prediction.py`: Implements the full retrieval pipeline, from initial BM25 retrieval to re-ranking using the trained model.
5.  `4b_eval_runs.ipynb`: Evaluates the generated run files against ground truth (qrels).
6.  `4c_transform_to_trec_run_format.ipynb`: Converts the system's output into the TREC run format for official evaluation.

## Step-by-Step Guide

This section details each component of the pipeline.

### Step 1: Topic Clustering (`1_fr_TermClustering_onebig_cluster.py`)

**Purpose**: To group similar topics together into clusters. This can help the model learn cluster-specific relevance patterns.

**Key Functionalities**:
*   Loads topics from the database.
*   Preprocesses topic text (tokenization, stemming, stopword removal).
*   Calculates term frequencies within topics.
*   Generates embeddings for topics (e.g., using OpenAI's `text-embedding-3-small`).
*   Performs dimensionality reduction on embeddings (UMAP).
*   Applies K-Means clustering to group topics based on their embeddings.
*   Determines the optimal number of clusters using silhouette scores.

**Inputs**:
*   Topics from the "Topic" table in the database.
*   (Optionally) OpenAI API key for generating new embeddings.
*   k-Cluster via Silhouette Score - graph during execution.

**Outputs**:
*   `topics_cluster_all_subcollections.csv`: A CSV file mapping each `queryid` to a `cluster` number.
*   `topic_embeddings_all_subcollections.csv` / `.npy`: Topic embeddings.
*   HTML visualizations of term frequency, UMAP embeddings, and silhouette scores.

**Usage**:
   Execute the script from the command line. Ensure database credentials and OpenAI API key (if needed) are correctly set within the script.
   ⚠️ Choose k-Cluster during Execution.

### Step 2: Training Set Creation (`2_fr_create_trainset_onebigCluster.py`)

**Purpose**: To prepare a dataset for training the neural re-ranking model. This involves processing documents associated with relevance judgments (qrels).

**Key Functionalities**:
*   Fetches documents from the database that have relevance judgments for specific sub-collections.
*   **Term Frequency Calculation**:
    *   Preprocesses document text (cleaning, lowercasing, accent removal, stemming with French SnowballStemmer, stopword removal).
    *   Calculates global term frequencies across the selected documents.
*   **Multi-Hot Encoding**:
    *   Selects the top N most frequent stemmed terms (e.g., top 10,000) to form a vocabulary.
    *   For each document, creates a multi-hot vector representing the presence/absence of these top N terms.
    *   Associates these document vectors with their relevance scores and the cluster of their corresponding topic (from Step 1).
*   Uses parallel processing for efficiency in text processing and SQL queries.

**Inputs**:
*   "Document", "Qrel", and "Topic" tables from the database.
*   `topics_cluster_all_subcollections.csv` (from Step 1).
*   Database credentials.

**Outputs**:
*   `top_terms_stemmed_all_subcollections.csv`: A list of the most frequent stemmed terms and their counts, forming the vocabulary.
*   `words_index_translationtable.csv`: A mapping from terms in the vocabulary to integer indices.
*   `bis2023-02_train_set_documents_top_terms_all_subcollections.csv` (and a compressed `.npz` version): The main training dataset, containing `docid`, multi-hot encoded `term_idx`, `cluster` (topic cluster), and `relevance`. The prefix `bis2023-02` indicates the data used for training goes up to sub-collection "2023-02".

**Usage**:
   Run the script. It will connect to the database, process documents, and generate the output files.

### Step 3: Model Training (`3_create_model_hyperparameter_ai-support.py`)

**Purpose**: To train a deep learning model that takes a document's multi-hot representation as input and predicts its relevance and cluster.

**Key Functionalities**:
*   Loads the training data generated in Step 2 (specifically the `.npz` file for efficiency).
*   Splits the data into training, validation, and test sets. Stratification is used based on cluster and relevance to ensure balanced splits.
*   Defines a Keras neural network model architecture:
    *   Input layer for the multi-hot encoded document vectors (e.g., 10,000 dimensions).
    *   Several dense layers with LeakyReLU activation, Batch Normalization, and Dropout for regularization.
    *   Two output branches:
        *   One for cluster prediction (softmax activation).
        *   One for relevance prediction (sigmoid activation).
*   Compiles the model with appropriate loss functions (categorical crossentropy for cluster, binary crossentropy for relevance) and metrics (accuracy, precision, recall, AUC for relevance).
*   Implements class weights for the relevance task to handle imbalanced data.
*   Uses callbacks for training:
    *   Learning rate scheduling (e.g., cosine annealing, ReduceLROnPlateau).
    *   Early stopping based on validation relevance precision.
    *   Model checkpointing to save the best model based on validation relevance precision.
*   Trains the model and evaluates it on the test set.

**Inputs**:
*   `train_set_documents_top_terms_all_subcollections.npz` (or similarly named file based on the `subset` variable, from Step 2). The script is designed to process "all_subcollections" as a single training dataset.

**Outputs**:
*   Trained Keras models saved in the `models/` directory (e.g., `models/best_bs512_model_callback_best_epoch_all_subcollections.keras`).
*   Console output of training progress and evaluation metrics.

**Usage**:
   Execute the script. It will load data, build the model, train it, and save the best performing version.

   The `subcollections` variable within the script (e.g., `subcollections = ["all_subcollections"]`) determines which training set is used.

### Step 4: Full Retrieval and Re-ranking (`4a_full_retrieval_prediction.py`)

**Purpose**: To perform end-to-end retrieval: fetch initial results using BM25, then use the trained neural model to re-rank these results. This script processes data for specified sub-collections.

**Key Functionalities**:
*   Iterates through specified `SUB_COLLECTIONS` (e.g., "2023-03", "2023-04").
*   For each sub-collection:
    *   Loads the corresponding PyTerrier index and topics.
    *   Loads pre-computed topic-cluster mappings (from Step 1).
    *   **Initial Retrieval**: Retrieves documents for batches of topics using BM25.
    *   **Document Preparation**:
        *   Loads the global vocabulary (`top_terms_stemmed_all_subcollections.csv` from Step 2).
        *   For retrieved documents, fetches their text from the database.
        *   Preprocesses document text (tokenization, stemming, etc.) and creates multi-hot encoded vectors based on the global vocabulary. This step is similar to parts of Step 2 but applied at prediction time.
    *   **Prediction**: Loads the trained Keras model (from Step 3) and predicts cluster and relevance scores for the encoded documents.
    *   **Re-ranking**: Modifies the initial BM25 scores. If a document's predicted cluster matches the topic's true cluster, its BM25 score is multiplied by the predicted relevance score. Otherwise, the relevance factor is effectively 1 (no change to BM25 score beyond normalization).
    *   Saves the re-ranked results to a CSV file.
*   Uses parallel processing for document encoding.

**Inputs**:
*   PyTerrier indexes for each sub-collection (e.g., `BASE_PATH/index/longeval-web-fr-2023-03-pyterrier`).
*   "Topic", "Qrel", "Document" tables from the database.
*   `topics_cluster_all_subcollections.csv` (from Step 1).
*   `top_terms_stemmed_all_subcollections.csv` (vocabulary from Step 2).
*   The trained Keras model (e.g., `models/best_bs512_model_callback_best_epoch_all_subcollections.keras` from Step 3).
*   `BASE_PATH/metadata.yml`.

**Outputs**:
*   For each sub-collection, a run file: `run_modelprediction_{SUB_COLLECTION}.csv`. This contains columns like `qid`, `docno`, `rank`, original `score_bm25`, new re-ranked `score`, predicted relevance, and cluster information.

**Usage**:
   Configure database credentials, `BASE_PATH`, and the list of `SUB_COLLECTIONS` in the `if __name__ == "__main__":` block. Then run the script:

### Step 5: Evaluation (`4b_eval_runs.ipynb`)

**Purpose**: To evaluate the performance of the generated run files using standard IR metrics.

**Key Functionalities (within the Jupyter Notebook)**:
*   Loads a specific run file (e.g., `run_modelprediction_2023-02.csv` from Step 4) or a baseline BM25 run.
*   Loads the corresponding qrels (relevance judgments) from the database or a qrels file for the specific sub-collection.
*   Uses PyTerrier's `pt.Experiment` to compare the run against the qrels.
*   Calculates and displays metrics like nDCG, MAP, Precision@10, bpref.

**Inputs**:
*   A run file (e.g., `run_modelprediction_{SUB_COLLECTION}.csv`).
*   Qrels for the corresponding sub-collection (from database or file).
*   PyTerrier index and topic data for context if needed.

**Outputs**:
*   Pandas DataFrames displaying evaluation metrics.

**Usage**:
   Open and run the cells in the Jupyter Notebook. Modify file paths and sub-collection names as needed to evaluate different runs.

### Step 6: Output Formatting (`4c_transform_to_trec_run_format.ipynb`)

**Purpose**: To convert the system's output run files into the standard TREC format, which is often required for shared task submissions.

**Key Functionalities (within the Jupyter Notebook)**:
*   Iterates through a list of run files (e.g., `submit_run_2023-02.csv`, which are assumed to be the outputs from Step 4, possibly renamed).
*   For each run file:
    *   Reads the CSV.
    *   Transforms the DataFrame into the TREC format:
      `qid Q0 docno rank score system_name`
      (where `Q0` is a literal, and `system_name` is derived from the sub-collection).
    *   Saves the TREC-formatted run to a new file (e.g., in a `submissions/{SUB_COLLECTION}/run.txt` structure).

**Inputs**:
*   Run files in CSV format (outputs of Step 4, potentially renamed as `submit_run_{SUB_COLLECTION}.csv`).

**Outputs**:
*   TREC-formatted run files (e.g., `submissions/2023-02/run.txt`).

**Usage**:
   Update the list of `runs` and `BASE_PATH` in the notebook, then run the cells.

## Configuration

Key configuration parameters are typically found at the beginning of the scripts or within the `if __name__ == "__main__":` blocks:

*   **`DATABASE`, `USER`, `HOST`, `PORT`, `PASSWORD`**: For PostgreSQL database connection.
*   **`DATASET`, `LANGUAGE`**: Dataset identifiers (e.g., "longeval-web", "fr").
*   **`SUB_COLLECTIONS`**: A list of sub-collection identifiers (e.g., `["2023-03", "2023-04"]`) that the pipeline will process.
*   **`BASE_PATH`**: The root directory for the LongEval dataset and associated files (indexes, metadata).
*   **Model paths and vocabulary file names**: Defined within the scripts, ensure these match your generated files.

## Adapting for Future Projects

This framework provides a solid backbone for developing neural retrieval systems. To adapt it:

*   **Data Sources**: Modify SQL queries and data loading functions (`load_topics`, `load_index`, data fetching in `process_batch_MHE`, etc.) to match your new dataset structure and format.
*   **Text Preprocessing**: Adjust `stem_text` and other NLP functions (stemmers, stopwords, tokenizers) if working with a different language or text domain.
*   **Feature Engineering**: The current system uses multi-hot encoding. You might explore other representations like TF-IDF, word embeddings (Word2Vec, FastText), or contextual embeddings from transformers (BERT, etc.) as input to your neural model. This would require significant changes in `2_fr_create_trainset_onebigCluster.py`, `3_create_model_hyperparameter_ai-support.py`, and `4a_full_retrieval_prediction.py`.
*   **Neural Model Architecture**: The `create_search_model` function in `3_create_model_hyperparameter_ai-support.py` can be replaced with any Keras model suitable for your task. You might change layer types, sizes, activation functions, or the overall structure.
*   **Clustering**: Topic clustering (Step 1) is optional or can be replaced by other methods of incorporating topic/query information. If not used, remove dependencies on cluster predictions in the model and re-ranking logic.
*   **Re-ranking Strategy**: The re-ranking logic in `4a_full_retrieval_prediction.py` currently combines BM25 scores with model predictions. This can be customized (e.g., using the model score directly, a learned combination, or more complex fusion methods).
*   **Evaluation**: Ensure your qrels and evaluation metrics in `4b_eval_runs.ipynb` are appropriate for your new task.

This Pipeline can be systematically build on top of new and effective neural retrieval systems by changing connections to exterior datasets.