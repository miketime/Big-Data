"""
Dimensionality Reduction for Car Dataset with NVIDIA GPU Acceleration
Optimized for NVIDIA 3050 Ti on Windows 11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Check for GPU and install required packages if necessary
def check_and_setup_gpu():
    """Check for CUDA GPU and set up required packages"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name}")
            print(f"CUDA Version: {torch.version.cuda}")
            return False
        else:
            print("No CUDA-compatible GPU detected. Will use CPU instead.")
            return False
    except ImportError:
        print("PyTorch not found. Installing required packages for GPU acceleration...")
        

# Load and preprocess data
def load_data(file_path):
    """Load and parse the car dataset CSV file"""
    print("Loading data...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def preprocess_data(df, use_gpu=False):
    """Preprocess the data for dimensionality reduction with GPU acceleration if available"""
    print("Preprocessing data...")
    
    # Select relevant features
    numerical_features = ['manufacturing_year', 'engine_hp', 'engine_cc', 'km_no', 'price', 'doors_no']
    categorical_features = ['make', 'fuel_type', 'transmission_type', 'body']
    
    # Select only the features we need
    selected_features = numerical_features + categorical_features
    df_selected = df[selected_features].copy()
    
    # Print data information
    print("Selected features:", selected_features)
    print("Dataset shape after feature selection:", df_selected.shape)
    
    # Check how many missing values we have
    missing_values = df_selected.isnull().sum()
    print("\nMissing values before imputation:")
    print(missing_values)
    
    # Replace "\N" with NaN
    df_selected.replace("\\N", np.nan, inplace=True)
    
    # Convert columns to appropriate types
    for feature in numerical_features:
        try:
            df_selected[feature] = pd.to_numeric(df_selected[feature], errors='coerce')
        except:
            pass
    
    if use_gpu:
        try:
            # GPU-accelerated preprocessing using RAPIDS
            import cudf
            import cuml
            from cuml.preprocessing import StandardScaler
            
            # First handle preprocessing on CPU
            # Handle missing values
            for feature in numerical_features:
                median_val = df_selected[feature].median()
                df_selected[feature].fillna(median_val, inplace=True)
            
            for feature in categorical_features:
                most_freq = df_selected[feature].mode()[0]
                df_selected[feature].fillna(most_freq, inplace=True)
            
            # Perform one-hot encoding for categorical features
            df_encoded = pd.get_dummies(df_selected, columns=categorical_features, dummy_na=False)
            
            # Move to GPU
            print("Converting to GPU dataframe...")
            df_gpu = cudf.DataFrame.from_pandas(df_encoded)
            
            # Scale numerical features on GPU
            print("Scaling features on GPU...")
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_gpu[df_encoded.columns])
            
            # Convert back to numpy for compatibility
            X_processed = scaled_features.to_pandas().values
            
            print(f"Processed data shape: {X_processed.shape}")
            return X_processed, df_selected, df_encoded.columns.tolist()
            
        except (ImportError, ModuleNotFoundError) as e:
            print(f"GPU preprocessing failed: {e}")
            print("Falling back to CPU preprocessing...")
            use_gpu = False
    
    if not use_gpu:
        # CPU preprocessing
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Apply preprocessing
        print("Applying preprocessing transformations on CPU...")
        X_processed = preprocessor.fit_transform(df_selected)
        
        # Get feature names after one-hot encoding
        onehot_features = []
        for i, feature in enumerate(categorical_features):
            categories = preprocessor.transformers_[1][1].named_steps['onehot'].categories_[i]
            for category in categories:
                onehot_features.append(f"{feature}_{category}")
                
        feature_names = numerical_features + onehot_features
        print(f"Processed data shape: {X_processed.shape}")
        
        return X_processed, df_selected, feature_names

# GPU-accelerated dimensionality reduction algorithms
def apply_pca(X, n_components=3, use_gpu=False):
    """Apply PCA dimensionality reduction with GPU acceleration if available"""
    print("\nApplying PCA...")
    start_time = time.time()
    
    if use_gpu:
        try:
            from cuml.decomposition import PCA as cuPCA
            print("Using GPU-accelerated PCA")
            pca = cuPCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            
            # Convert to numpy array
            if hasattr(X_pca, 'to_pandas'):
                X_pca = X_pca.to_pandas().values
            elif hasattr(X_pca, 'get'):
                X_pca = X_pca.get()
                
            end_time = time.time()
            print(f"PCA completed in {end_time - start_time:.2f} seconds (GPU)")
            print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
            
            return X_pca, pca
            
        except (ImportError, ModuleNotFoundError) as e:
            print(f"GPU PCA failed: {e}")
            print("Falling back to CPU PCA...")
            use_gpu = False
    
    if not use_gpu:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        end_time = time.time()
        
        print(f"PCA completed in {end_time - start_time:.2f} seconds (CPU)")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
        
        return X_pca, pca

def apply_tsne(X, n_components=3, perplexity=30, n_iter=500, use_gpu=False):
    """Apply t-SNE dimensionality reduction with GPU acceleration if available"""
    print("\nApplying t-SNE...")
    start_time = time.time()
    
    if use_gpu:
        try:
            from cuml.manifold import TSNE as cuTSNE
            print("Using GPU-accelerated t-SNE")
            tsne = cuTSNE(n_components=n_components, perplexity=perplexity, 
                        n_iter=n_iter, random_state=42)
            X_tsne = tsne.fit_transform(X)
            
            # Convert to numpy array
            if hasattr(X_tsne, 'to_pandas'):
                X_tsne = X_tsne.to_pandas().values
            elif hasattr(X_tsne, 'get'):
                X_tsne = X_tsne.get()
                
            end_time = time.time()
            print(f"t-SNE completed in {end_time - start_time:.2f} seconds (GPU)")
            
            return X_tsne
            
        except (ImportError, ModuleNotFoundError) as e:
            print(f"GPU t-SNE failed: {e}")
            print("Falling back to CPU t-SNE...")
            use_gpu = False
    
    if not use_gpu:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                   n_iter=n_iter, random_state=42)
        X_tsne = tsne.fit_transform(X)
        end_time = time.time()
        
        print(f"t-SNE completed in {end_time - start_time:.2f} seconds (CPU)")
        
        return X_tsne

def apply_umap_reduction(X, n_components=3, n_neighbors=15, min_dist=0.1, use_gpu=False):
    """Apply UMAP dimensionality reduction with GPU acceleration if available"""
    print("\nApplying UMAP...")
    start_time = time.time()
    
    if use_gpu:
        try:
            from cuml.manifold import UMAP as cuUMAP
            print("Using GPU-accelerated UMAP")
            umap_model = cuUMAP(n_components=n_components, n_neighbors=n_neighbors, 
                             min_dist=min_dist, random_state=42)
            X_umap = umap_model.fit_transform(X)
            
            # Convert to numpy array
            if hasattr(X_umap, 'to_pandas'):
                X_umap = X_umap.to_pandas().values
            elif hasattr(X_umap, 'get'):
                X_umap = X_umap.get()
                
            end_time = time.time()
            print(f"UMAP completed in {end_time - start_time:.2f} seconds (GPU)")
            
            return X_umap, umap_model
            
        except (ImportError, ModuleNotFoundError) as e:
            print(f"GPU UMAP failed: {e}")
            print("Falling back to CPU UMAP...")
            use_gpu = False
    
    if not use_gpu:
        import umap
        umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                           min_dist=min_dist, random_state=42)
        X_umap = umap_model.fit_transform(X)
        end_time = time.time()
        
        print(f"UMAP completed in {end_time - start_time:.2f} seconds (CPU)")
        
        return X_umap, umap_model

# Visualization functions
def visualize_2d(X_reduced, df_original, title, colorby='price', algo_name=''):
    """Create a 2D visualization of the reduced data"""
    plt.figure(figsize=(10, 8))
    
    # Color by the specified column if it exists
    if colorby in df_original.columns:
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                              c=df_original[colorby], cmap='viridis', 
                              alpha=0.5, s=50)
        plt.colorbar(scatter, label=colorby)
    else:
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5, s=50)
    
    plt.title(f'{title} - Colored by {colorby}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(f"{algo_name}_2d_{colorby}.png")
    print(f"Saved 2D visualization to {algo_name}_2d_{colorby}.png")

def visualize_3d(X_reduced, df_original, title, colorby='price', algo_name=''):
    """Create a 3D visualization of the reduced data"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by the specified column if it exists
    if colorby in df_original.columns:
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],
                            c=df_original[colorby], cmap='viridis', 
                            alpha=0.5, s=50)
        plt.colorbar(scatter, label=colorby)
    else:
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], alpha=0.5, s=50)
    
    ax.set_title(f'{title} - Colored by {colorby}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.tight_layout()
    plt.savefig(f"{algo_name}_3d_{colorby}.png")
    print(f"Saved 3D visualization to {algo_name}_3d_{colorby}.png")

# Main function
def main(file_path, n_components=3):
    """Main function to execute the dimensionality reduction process with GPU acceleration if available"""
    # Check for GPU
    use_gpu = check_and_setup_gpu()
    
    # Load data
    df = load_data(file_path)
    
    # Preprocess data
    X_processed, df_selected, feature_names = preprocess_data(df, use_gpu)
    
    # Apply PCA
    X_pca, pca_model = apply_pca(X_processed, n_components, use_gpu)
    
    # Apply t-SNE
    X_tsne = apply_tsne(X_processed, n_components, 
                       perplexity=min(30, len(df_selected)//5), 
                       use_gpu=use_gpu)
    
    # Apply UMAP
    X_umap, umap_model = apply_umap_reduction(X_processed, n_components, use_gpu=use_gpu)
    
    # Visualize results - 2D and 3D
    # PCA
    visualize_2d(X_pca[:, :2], df_selected, 'PCA', 'price', 'pca')
    visualize_2d(X_pca[:, :2], df_selected, 'PCA', 'manufacturing_year', 'pca')
    visualize_3d(X_pca, df_selected, 'PCA', 'price', 'pca')
    
    # t-SNE
    visualize_2d(X_tsne[:, :2], df_selected, 't-SNE', 'price', 'tsne')
    visualize_2d(X_tsne[:, :2], df_selected, 't-SNE', 'manufacturing_year', 'tsne')
    visualize_3d(X_tsne, df_selected, 't-SNE', 'price', 'tsne')
    
    # UMAP
    visualize_2d(X_umap[:, :2], df_selected, 'UMAP', 'price', 'umap')
    visualize_2d(X_umap[:, :2], df_selected, 'UMAP', 'manufacturing_year', 'umap')
    visualize_3d(X_umap, df_selected, 'UMAP', 'price', 'umap')
    
    # Return the reduced data and models
    return {
        'pca': {'data': X_pca, 'model': pca_model},
        'tsne': {'data': X_tsne},
        'umap': {'data': X_umap, 'model': umap_model},
        'original_data': df_selected,
        'feature_names': feature_names
    }

if __name__ == "__main__":
    # Replace with the actual path to your file
    file_path = "market_data.csv"
    results = main(file_path)
    
    print("\nDimensionality reduction complete!")