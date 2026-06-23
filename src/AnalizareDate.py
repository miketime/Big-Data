import pandas as pd
import numpy as np
import csv

# Load sample data
sample = pd.read_csv('market_data.csv', 
                     nrows=5000,
                     delimiter=',', 
                     quotechar='"', 
                     quoting=csv.QUOTE_ALL)

# Analyze object columns
object_cols = sample.select_dtypes(include=['object']).columns.tolist()
numeric_cols = sample.select_dtypes(include=['number']).columns.tolist()

print(f"Numeric columns: {numeric_cols}")

# Check cardinality (number of unique values) for each object column
cardinality = {}
for col in object_cols:
    unique_count = sample[col].nunique()
    cardinality[col] = unique_count
    print(f"Column '{col}': {unique_count} unique values")

# Group columns by cardinality
low_cardinality = [col for col, count in cardinality.items() if count <= 10]
medium_cardinality = [col for col, count in cardinality.items() if 10 < count <= 100]
high_cardinality = [col for col, count in cardinality.items() if count > 100]

print(f"Low cardinality columns: {len(low_cardinality)}")
print(f"Medium cardinality columns: {len(medium_cardinality)}")
print(f"High cardinality columns: {len(high_cardinality)}")

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import scipy.sparse as sp

# Initialize transformers for different column types
preprocessor = ColumnTransformer(
    transformers=[
        # One-hot encode low cardinality columns
        ('one_hot', OneHotEncoder(sparse=True, handle_unknown='ignore'), low_cardinality),
        
        # Label encode medium cardinality columns
        ('label', Pipeline([
            ('label_encoder', LabelEncoder())
        ]), medium_cardinality),
        
        # For high cardinality, use count vectorization or other techniques
        # This depends on the nature of these columns (text vs identifiers)
        # Example for text-like columns:
        ('count_vec', CountVectorizer(max_features=100), high_cardinality),
        
        # Pass through numeric columns
        ('numeric', 'passthrough', numeric_cols)
    ],
    remainder='drop'
)

# Process in chunks
chunk_size = 10000
processed_chunks = []

for i, chunk in enumerate(pd.read_csv('your_large_file.csv', 
                                     chunksize=chunk_size,
                                     delimiter=',', 
                                     quotechar='"', 
                                     quoting=csv.QUOTE_ALL)):
    
    print(f"Processing chunk {i+1}")
    
    # Apply preprocessing to chunk
    if i == 0:
        # Fit and transform first chunk
        transformed = preprocessor.fit_transform(chunk)
    else:
        # Transform subsequent chunks
        transformed = preprocessor.transform(chunk)
    
    processed_chunks.append(transformed)
    
    # Optional: Save processed chunk to disk to save memory
    # sp.save_npz(f'processed_chunk_{i}.npz', transformed)
    # processed_chunks = []  # Clear memory

# Combine processed chunks (if memory allows)
# Or process them incrementally with IncrementalPCA

# Apply dimension reduction
svd = TruncatedSVD(n_components=20)  # Adjust components as needed
reduced_data = svd.fit_transform(sp.vstack(processed_chunks))