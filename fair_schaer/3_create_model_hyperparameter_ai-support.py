#!/usr/bin/env python
# coding: utf-8

# # 1. MultiHotEncoding

# In[1]:


import os
import ast # lib for converting list saved as string back to list object
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
#from sqlalchemy import create_engine


# In[2]:


# SQL Connection
DATABASE = "longeval"
USER = "dis18"
HOST = "db"
PORT = "5432"
PASSWORD = "dis182425"

#engine = create_engine(f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")

#df = pd.read_sql('select * from "Topic" limit 1', con=engine)
#sql_query = lambda x: pd.read_sql(x, con=engine)


# In[3]:


#test_df = pd.read_csv("train_set_documents_top_terms_2022-06.csv")
#print(test_df)
#print(test_df["cluster"].max())


# In[4]:


# Prepare data
def get_subsetdata(subset: str = None)-> pd.DataFrame:
    """
    Load DataFrame from NumPy format with lists preserved.
    
    Parameters:
        filename: Filename without extension
        
    Returns:
        DataFrame with all original data types preserved
    """
    # Load data from NPZ file
    data = np.load(f"train_set_documents_top_terms_{subset}.npz", allow_pickle=True)
    
    
    # Create dictionary for DataFrame constructor
    df_dict = {}
    
    # Track detected types for verification
    detected_types = {}
    saved_types = {}
    
    # Check if we have saved type information
    has_type_info = '__column_types__' in data.files
    if has_type_info:
        saved_types = data['__column_types__'][0]
        print(f"Found saved type information for {len(saved_types)} columns")
    
    # Process each column
    for key in data.files:
        if key == '__column_types__':
            continue  # Skip metadata
            
        # Get the array
        arr = data[key]
        
        # Detect the type regardless of saved info (for verification)
        if arr.dtype == np.dtype('O'):  # Object dtype
            # Sample non-None values
            samples = [x for x in arr[:min(10, len(arr))] if x is not None]
            
            if len(samples) > 0:
                sample = samples[0]
                
                # Check if it's likely a list
                if isinstance(sample, np.ndarray) or (hasattr(sample, '__iter__') and not isinstance(sample, (str, bytes))):
                    detected_types[key] = 'list'
                    # Convert numpy arrays to Python lists
                    df_dict[key] = [x.tolist() if isinstance(x, np.ndarray) else x for x in arr]
                else:
                    # It's some other object type
                    detected_types[key] = 'object'
                    df_dict[key] = arr
            else:
                # No valid samples, keep as is
                detected_types[key] = 'object'
                df_dict[key] = arr
        elif np.issubdtype(arr.dtype, np.integer):
            detected_types[key] = 'int'
            df_dict[key] = arr
        elif np.issubdtype(arr.dtype, np.floating):
            detected_types[key] = 'float'
            df_dict[key] = arr
        else:
            # Other numpy dtype
            detected_types[key] = str(arr.dtype)
            df_dict[key] = arr
    
    # Create initial DataFrame
    df = pd.DataFrame(df_dict)
    
    # Verify and apply additional type corrections
    for col in df.columns:
        # Apply saved type information but verify it's correct
        if has_type_info and col in saved_types:
            saved_type = saved_types[col]
            detected_type = detected_types.get(col, 'unknown')
            
            if saved_type != detected_type:
                print(f"Warning: Column '{col}' has saved type '{saved_type}' but detected as '{detected_type}'")
                # Try to reconcile differences
                
                # Special case: strings that should be lists
                if saved_type == 'list' and detected_type != 'list':
                    # Check if strings that look like lists
                    if isinstance(df[col].iloc[0], str) and df[col].iloc[0].startswith('['):
                        try:
                            df[col] = df[col].apply(ast.literal_eval)
                            print(f"  Fixed: Converted string representations to lists in '{col}'")
                        except (ValueError, SyntaxError):
                            print(f"  Failed: Could not convert strings to lists in '{col}'")
                
                # Special case: strings that should be numeric
                elif saved_type in ('int', 'float') and detected_type not in ('int', 'float'):
                    try:
                        if saved_type == 'int':
                            df[col] = df[col].astype(int)
                            print(f"  Fixed: Converted to integers in '{col}'")
                        else:
                            df[col] = df[col].astype(float)
                            print(f"  Fixed: Converted to floats in '{col}'")
                    except (ValueError, TypeError):
                        print(f"  Failed: Could not convert to {saved_type} in '{col}'")
        
        # For all columns, check for string representations that need conversion
        if isinstance(df[col].iloc[0], str):
            # Check for string representation of lists
            if df[col].iloc[0].startswith('[') and df[col].iloc[0].endswith(']'):
                try:
                    df[col] = df[col].apply(ast.literal_eval)
                    print(f"Converted string representations to lists in column '{col}'")
                except (ValueError, SyntaxError):
                    pass  # Not valid list representations
            
            # Check for numeric strings not already converted
            elif col not in saved_types or saved_types[col] not in ('int', 'float'):
                # Try converting to integer
                try:
                    df[col] = df[col].astype(int)
                    print(f"Converted strings to integers in column '{col}'")
                except (ValueError, TypeError):
                    # Try converting to float
                    try:
                        df[col] = df[col].astype(float)
                        print(f"Converted strings to floats in column '{col}'")
                    except (ValueError, TypeError):
                        pass  # Not numeric strings
    df.loc[df["relevance"]>0, ["relevance"]]=1
    return df[["docid", "term_idx", "relevance", "cluster"]]


# In[5]:


# Split Train, Val, Test
# Source: Help with ClaudeAI
def train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ 
    Returns sliced Data Frames for Train, Val, Test.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Split proportions must sum to 1")

    # Calculate sizes for each split
    n_samples = len(df)
    n_train = int(n_samples * train_size)
    n_val = int(n_samples * val_size)
    
    # Create random indices for splitting
    indices = np.random.permutation(n_samples) # -> So cool!!! I need to remember that!!!
        
    # Simple random split
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

     # Create the splits
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


# In[6]:


# Vectorize the term idx into a multihotencoded tensor
# Source: DIS21a.1 Heisenberg
# Remark: Normal OneHotEncoding is not possible because its in fact a multi Hot Encoding.

# just take the first 10.000 most frequent words
def vectorize_sequences(sequences, dimension=10_000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences): # i is the n_th review whereas the sequence assigns like a list of fields (like pandas data frame) all the right fields results[i, [3, 5]] = 1
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


# # 2. Train Subset Model

# In[7]:


# Sources
# https://medium.com/@kevinnjagi83/building-deep-learning-models-with-multi-output-architectures-61d1c3c81d40
# https://medium.com/@sthanikamsanthosh1994/custom-models-with-tensorflow-part-1-multi-output-model-c01a78e67d47
# ClaudeAI (altered, because no one explained how to prepare labels...)

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

def get_label(df):
    """
    Returns multilabel varaiable for Dense Layer Training.
    """
    cluster_labels = np.array(df["cluster"].tolist())
    relevance_labels = np.array(df["relevance"].tolist())
    
    cluster_encoded = to_categorical(cluster_labels)
    relevance_formated = relevance_labels.astype("float32")

    # Remark: Cluster Shape is not word shape... Got confused for a sec. xD
    print("Cluster-Shape: ",cluster_encoded.shape, "\tRelevance-Shape: ", relevance_formated.shape)

    return [cluster_encoded, relevance_formated]


# In[8]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC

def create_search_model(
    text_input_shape,  # Shape of text features
    num_clusters      # Number of clusters
):
    # Single input for text
    text_input = Input(shape=text_input_shape, name='text_input')
    
    # First layer - increased capacity
    x = Dense(512, kernel_regularizer=l2(0.0005))(text_input)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Second layer
    x = Dense(256, kernel_regularizer=l2(0.0005))(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    
    # Shared layer before branching
    x = Dense(128, kernel_regularizer=l2(0.0005))(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = BatchNormalization()(x)
    shared = Dropout(0.3)(x)
    
    # Branch for cluster task
    cluster_x = Dense(96, kernel_regularizer=l2(0.0005))(shared)
    cluster_x = LeakyReLU(negative_slope=0.1)(cluster_x)
    cluster_x = BatchNormalization()(cluster_x)
    cluster_x = Dropout(0.25)(cluster_x)
    cluster_x = Dense(64, kernel_regularizer=l2(0.0005))(cluster_x)
    cluster_x = LeakyReLU(negative_slope=0.1)(cluster_x)
    cluster_output = Dense(num_clusters, activation='softmax', name='cluster')(cluster_x)
    
    # Branch for binary relevance task
    relevance_x = Dense(96, kernel_regularizer=l2(0.0005))(shared)
    relevance_x = LeakyReLU(negative_slope=0.1)(relevance_x)
    relevance_x = BatchNormalization()(relevance_x)
    relevance_x = Dropout(0.4)(relevance_x)
    relevance_x = Dense(48, kernel_regularizer=l2(0.0005))(relevance_x)
    relevance_x = LeakyReLU(negative_slope=0.1)(relevance_x)
    relevance_x = BatchNormalization()(relevance_x)
    relevance_x = Dropout(0.35)(relevance_x)
    
    # Output layer for binary relevance
    relevance_output = Dense(1, activation='sigmoid', name='relevance')(relevance_x)
    
    # Create model
    model = Model(
        inputs=text_input,
        outputs=[cluster_output, relevance_output]
    )
    
    # Compile with standard losses
    model.compile(
        optimizer=Adam(learning_rate=0.0008),
        loss={
            'cluster': 'categorical_crossentropy',
            'relevance': 'binary_crossentropy'
        },
        loss_weights={
            'cluster': 0.35,
            'relevance': 1.0
        },
        metrics={
            'cluster': ['accuracy'],
            'relevance': ['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
        }
    )
    
    return model


# In[1]:


import os
import gc # memory efficiency
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau

def train_model(subset=None, df_train=None, n_words=None):
    """
    Improved training function with better learning rate scheduling and data handling
    """

    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Split data with stratification for both cluster and relevance
    # First convert relevance to string to use in stratification
    df_train['strat_col'] = df_train['cluster'].astype(str) + '_' + df_train['relevance'].astype(str)
    
    train_df, temp_df = train_test_split(
        df_train, test_size=0.3, stratify=df_train['strat_col'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['strat_col'], random_state=42
    )

    # Remove Data Frames after splitting data in sets
    del df_train, temp_df
    gc.collect()
    
    # Clean up the stratification column
    train_df.drop('strat_col', axis=1, inplace=True)
    val_df.drop('strat_col', axis=1, inplace=True)
    test_df.drop('strat_col', axis=1, inplace=True)
    
    print(f"Data split - Train: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")

    # Vectorize data
    x_train = vectorize_sequences(train_df["term_idx"], dimension=n_words)
    x_val = vectorize_sequences(val_df["term_idx"], dimension=n_words)
    x_test = vectorize_sequences(test_df["term_idx"], dimension=n_words)
    
    # Get labels
    y_train = get_label(train_df)
    y_val = get_label(val_df)
    y_test = get_label(test_df)
    
    # Calculate class weights for relevance with more emphasis on positive class
    relevance_class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_df["relevance"]),
        y=train_df["relevance"]
    )
    # Adjust weights to prioritize precision (penalize false positives more)
    relevance_class_weights[1] = relevance_class_weights[1] * 1.5  # Increase weight for positive class
    relevance_weight_dict = {i: relevance_class_weights[i] for i in range(len(relevance_class_weights))}
    print(f"Relevance class weights: {relevance_weight_dict}")
    
    # Create model
    text_shape = (n_words,)
    num_clusters = len(y_train[0][0])
    model = create_search_model(text_shape, num_clusters)
    
    # Set up callbacks
    def cosine_annealing_lr(epoch, initial_lr=0.0008, min_lr=1e-6, total_epochs=100):
        """Cosine annealing learning rate schedule"""
        import numpy as np
        return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / total_epochs)) / 2
    
    callbacks = [
        # Cosine Annealing learning rate scheduler
        LearningRateScheduler(cosine_annealing_lr),
        
        # Reduce LR on plateau as backup
        ReduceLROnPlateau(
            monitor='val_relevance_precision',
            factor=0.6,
            patience=4,
            min_lr=1e-6,
            verbose=1,
            mode='max'  # We want to maximize precision
        ),
        
        # Early stopping based on validation precision
        EarlyStopping(
            monitor='val_relevance_precision',
            patience=12,
            restore_best_weights=True,
            verbose=1,
            mode='max',  # We want to maximize precision
            min_delta=0.001
        ),
        
        # Checkpoint based on validation precision
        ModelCheckpoint(
            f'models/model_callback_best_epoch_{subset}.keras',
            monitor='val_relevance_precision',
            save_best_only=True,
            mode='max',  # We want to maximize precision
            verbose=1
        )
    ]

    # Training parameters
    epochs = 100  # Early stopping will determine actual epochs
    batch_size = 640 # original: 128
    
    # Create sample weights for relevance task
    sample_weights = np.ones(len(train_df))
    for i, rel in enumerate(train_df["relevance"]):
        sample_weights[i] = relevance_weight_dict[int(rel)]

    # Remove Data Frames as vectorized data exists
    del train_df, val_df, test_df
    gc.collect()
    
    # Train model
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        sample_weight=sample_weights
    )
    print("Training complete!")
    
    # Save model with a standard format
    model_name = f"standard_{subset}"
    model.save(f'models/{model_name}.keras')
    
    # Also save model using SavedModel format which preserves the entire model
    tf.keras.models.save_model(model, f'models/model_earlystop_callback_{model_name}.keras')
    
    # Evaluate on test set with focus on precision metrics
    results = model.evaluate(x_test, y_test, batch_size=batch_size)
    metric_names = model.metrics_names
    
    for i, metric_name in enumerate(metric_names):
        print(f"{metric_name}: {results[i]:.4f}")
    
    # Get predictions
    predictions = model.predict(x_test)
    
    # Compute F1 score for cluster predictions
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # Get the most likely cluster
    y_true_cluster = np.argmax(y_test[0], axis=1)
    y_pred_cluster = np.argmax(predictions[0], axis=1)
    
    # Compute macro and weighted F1
    cluster_f1_macro = f1_score(y_true_cluster, y_pred_cluster, average='macro')
    cluster_f1_weighted = f1_score(y_true_cluster, y_pred_cluster, average='weighted')
    
    print(f"Cluster F1 (macro): {cluster_f1_macro:.4f}")
    print(f"Cluster F1 (weighted): {cluster_f1_weighted:.4f}")
    
    # Compute precision and recall for relevance
    y_true_relevance = y_test[1]
    y_pred_relevance = (predictions[1] > 0.5).astype(int).flatten()
    
    relevance_precision = precision_score(y_true_relevance, y_pred_relevance)
    relevance_recall = recall_score(y_true_relevance, y_pred_relevance)
    relevance_f1 = f1_score(y_true_relevance, y_pred_relevance)
    
    print(f"Relevance Precision: {relevance_precision:.4f}")
    print(f"Relevance Recall: {relevance_recall:.4f}")
    print(f"Relevance F1: {relevance_f1:.4f}")
    
    return model, history


# In[10]:


def main(subset: str = None, n_words=None):
    df_train = get_subsetdata(subset=subset)
    print(df_train)
    df_train["term_idx"] = df_train["term_idx"].apply(lambda x: list(x))
    if not df_train.empty:
        train_model(subset=subset, df_train=df_train, n_words=n_words)


# In[11]:


# Get sub_collection and count(*) for each
#query= """
#select a.sub_collection, count(*)
#from "Document" a
#group by a.sub_collection
#"""
#df_subcol_count = sql_query(query)
#print(df_subcol_count)
#subcollections = df_subcol_count["sub_collection"].tolist()

subcollections = ["all_subcollections"] #['2022-06', '2022-07', '2022-09', '2023-01', '2023-06', '2023-08']


# In[12]:


for idx, subcollection in enumerate(subcollections, start=1):
    print("Creating Model for: ", subcollection)
    main(subset=subcollection, n_words=10_000)

print("\n!!!DONE!!!")


# In[ ]:




