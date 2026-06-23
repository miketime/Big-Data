import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Standard CPU implementations (as fallback)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA as CPU_PCA
from sklearn.manifold import TSNE as CPU_TSNE
import umap as cpu_umap

# GPU implementations
try:
    import cudf
    import cuml
    import cupy as cp
    from cuml.decomposition import PCA as GPU_PCA
    from cuml.manifold import TSNE as GPU_TSNE
    from cuml.manifold import UMAP as GPU_UMAP
    
    # Verify GPU is available
    print("NVIDIA GPU detected and RAPIDS libraries successfully imported")
    USE_GPU = True
    
    # Get GPU information
    import subprocess
    gpu_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
    print("GPU Information:")
    for line in gpu_info.split('\n'):
        if "NVIDIA" in line or "%" in line:  # Print relevant lines
            print(line)
except ImportError:
    print("Warning: RAPIDS libraries not found. Falling back to CPU implementations.")
    print("To use GPU acceleration, install RAPIDS: https://rapids.ai/start.html")
    USE_GPU = False

def load_data(file_path):
    """
    Load and parse the car dataset CSV file
    """
    print("Loading data...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def preprocess_data(df):
    """
    Preprocess the data for dimensionality reduction:
    - Select relevant features
    - Handle missing values
    - Convert categorical variables to numerical
    """
    print("Preprocessing data...")
    
    # Select relevant features - numerical and categorical that seem meaningful
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
    
    # Replace "\N" (which may be in strings) with NaN
    df_selected.replace("\\N", np.nan, inplace=True)
    
    # Convert columns to appropriate types
    for feature in numerical_features:
        try:
            df_selected[feature] = pd.to_numeric(df_selected[feature], errors='coerce')
        except:
            pass
    
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
    print("Applying preprocessing transformations...")
    X_processed = preprocessor.fit_transform(df_selected)
    
    # Get feature names after one-hot encoding
    onehot_features = []
    for i, feature in enumerate(categorical_features):
        categories = preprocessor.transformers_[1][1].named_steps['onehot'].categories_[i]
        for category in categories:
            onehot_features.append(f"{feature}_{category}")
            
    feature_names = numerical_features + onehot_features
    print(f"Processed data shape: {X_processed.shape}")
    
    # Return both the preprocessed data and original data for reference
    return X_processed, df_selected, feature_names

def apply_pca(X, n_components=3):
    """
    Apply PCA dimensionality reduction using GPU if available
    """
    print("\nApplying PCA...")
    
    if USE_GPU:
        # Convert to CuPy array for GPU
        try:
            X_gpu = cp.array(X)
            
            start_time = time.time()
            pca = GPU_PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_gpu)
            # Convert back to numpy for visualization
            X_pca = X_pca.get() if hasattr(X_pca, 'get') else X_pca
            end_time = time.time()
            
            print(f"GPU PCA completed in {end_time - start_time:.2f} seconds")
            print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
            
            return X_pca, pca
        except Exception as e:
            print(f"GPU PCA failed with error: {e}")
            print("Falling back to CPU implementation")
    
    # CPU fallback
    start_time = time.time()
    pca = CPU_PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    end_time = time.time()
    
    print(f"CPU PCA completed in {end_time - start_time:.2f} seconds")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    return X_pca, pca

def apply_tsne(X, n_components=3, perplexity=30, n_iter=500):
    """
    Apply t-SNE dimensionality reduction using GPU if available
    """
    print("\nApplying t-SNE...")
    
    if USE_GPU:
        try:
            # Convert to GPU array
            X_gpu = cp.array(X)
            
            start_time = time.time()
            # Note: cuML's TSNE has different parameter names
            tsne = GPU_TSNE(n_components=n_components, perplexity=perplexity, 
                           n_iter=n_iter, random_state=42)
            X_tsne = tsne.fit_transform(X_gpu)
            # Convert back to numpy for visualization
            X_tsne = X_tsne.get() if hasattr(X_tsne, 'get') else X_tsne
            end_time = time.time()
            
            print(f"GPU t-SNE completed in {end_time - start_time:.2f} seconds")
            return X_tsne
        except Exception as e:
            print(f"GPU t-SNE failed with error: {e}")
            print("Falling back to CPU implementation")
    
    # CPU fallback
    start_time = time.time()
    tsne = CPU_TSNE(n_components=n_components, perplexity=perplexity, 
                   n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X)
    end_time = time.time()
    
    print(f"CPU t-SNE completed in {end_time - start_time:.2f} seconds")
    
    return X_tsne

def apply_umap_reduction(X, n_components=3, n_neighbors=15, min_dist=0.1):
    """
    Apply UMAP dimensionality reduction using GPU if available
    """
    print("\nApplying UMAP...")
    
    if USE_GPU:
        try:
            # Convert to GPU array
            X_gpu = cp.array(X)
            
            start_time = time.time()
            umap_model = GPU_UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                               min_dist=min_dist, random_state=42)
            X_umap = umap_model.fit_transform(X_gpu)
            # Convert back to numpy for visualization
            X_umap = X_umap.get() if hasattr(X_umap, 'get') else X_umap
            end_time = time.time()
            
            print(f"GPU UMAP completed in {end_time - start_time:.2f} seconds")
            
            return X_umap, umap_model
        except Exception as e:
            print(f"GPU UMAP failed with error: {e}")
            print("Falling back to CPU implementation")
    
    # CPU fallback
    start_time = time.time()
    umap_model = cpu_umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                           min_dist=min_dist, random_state=42)
    X_umap = umap_model.fit_transform(X)
    end_time = time.time()
    
    print(f"CPU UMAP completed in {end_time - start_time:.2f} seconds")
    
    return X_umap, umap_model

def visualize_2d(X_reduced, df_original, title, colorby='price', algo_name=''):
    """
    Create a 2D visualization of the reduced data
    """
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
    """
    Create a 3D visualization of the reduced data
    """
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

def generate_performance_comparison(cpu_times, gpu_times):
    """
    Generate a performance comparison chart showing CPU vs GPU times
    """
    plt.figure(figsize=(12, 6))
    algorithms = list(cpu_times.keys())
    
    # Create comparison bar chart
    x = np.arange(len(algorithms))
    width = 0.35
    
    cpu_values = [cpu_times[algo] for algo in algorithms]
    gpu_values = [gpu_times[algo] for algo in algorithms]
    
    plt.bar(x - width/2, cpu_values, width, label='CPU Time')
    plt.bar(x + width/2, gpu_values, width, label='GPU Time')
    
    plt.xlabel('Algorithm')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison: CPU vs GPU')
    plt.xticks(x, algorithms)
    plt.legend()
    
    # Add speedup labels
    for i, (cpu, gpu) in enumerate(zip(cpu_values, gpu_values)):
        if gpu > 0:  # Avoid division by zero
            speedup = cpu / gpu
            plt.text(i, max(cpu, gpu) + 0.5, f'{speedup:.1f}x faster', 
                     ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("performance_comparison.png")
    print("Saved performance comparison to performance_comparison.png")

def main(file_path, n_components=3):
    """
    Main function to execute the dimensionality reduction process
    """
    # Load data
    df = load_data(file_path)
    
    # Preprocess data
    X_processed, df_selected, feature_names = preprocess_data(df)
    
    # Performance tracking
    cpu_times = {}
    gpu_times = {}
    
    # Apply PCA with timing
    start_time = time.time()
    X_pca, pca_model = apply_pca(X_processed, n_components)
    algorithm_time = time.time() - start_time
    
    if USE_GPU:
        gpu_times['PCA'] = algorithm_time
    else:
        cpu_times['PCA'] = algorithm_time
    
    # Apply t-SNE with timing
    start_time = time.time()
    X_tsne = apply_tsne(X_processed, n_components, perplexity=min(30, len(df_selected)//5))
    algorithm_time = time.time() - start_time
    
    if USE_GPU:
        gpu_times['t-SNE'] = algorithm_time
    else:
        cpu_times['t-SNE'] = algorithm_time
    
    # Apply UMAP with timing
    start_time = time.time()
    X_umap, umap_model = apply_umap_reduction(X_processed, n_components)
    algorithm_time = time.time() - start_time
    
    if USE_GPU:
        gpu_times['UMAP'] = algorithm_time
    else:
        cpu_times['UMAP'] = algorithm_time
    
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
    
    # If we have both CPU and GPU times, generate a comparison
    if USE_GPU:
        # For a fair comparison, run CPU versions too
        print("\nRunning CPU implementations for comparison...")
        
        # CPU PCA
        start_time = time.time()
        cpu_pca = CPU_PCA(n_components=n_components)
        _ = cpu_pca.fit_transform(X_processed)
        cpu_times['PCA'] = time.time() - start_time
        
        # CPU t-SNE
        start_time = time.time()
        cpu_tsne = CPU_TSNE(n_components=n_components, perplexity=min(30, len(df_selected)//5), random_state=42)
        _ = cpu_tsne.fit_transform(X_processed)
        cpu_times['t-SNE'] = time.time() - start_time
        
        # CPU UMAP
        start_time = time.time()
        cpu_umap_model = cpu_umap.UMAP(n_components=n_components, random_state=42)
        _ = cpu_umap_model.fit_transform(X_processed)
        cpu_times['UMAP'] = time.time() - start_time
        
        # Generate comparison chart
        generate_performance_comparison(cpu_times, gpu_times)
    
    # Return the reduced data and models
    return {
        'pca': {'data': X_pca, 'model': pca_model},
        'tsne': {'data': X_tsne},
        'umap': {'data': X_umap, 'model': umap_model},
        'original_data': df_selected,
        'feature_names': feature_names,
        'performance': {
            'cpu_times': cpu_times,
            'gpu_times': gpu_times
        }
    }

if __name__ == "__main__":
    # Replace with the actual path to your file
    file_path = "market_data.csv"
    
    print(f"Running dimensionality reduction on {'GPU' if USE_GPU else 'CPU'}")
    results = main(file_path)
    
    print("\nDimensionality reduction complete!")
    
    # Print performance summary
    if results['performance']['gpu_times']:
        print("\nPerformance Summary (GPU):")
        for algo, time_taken in results['performance']['gpu_times'].items():
            print(f"  {algo}: {time_taken:.2f} seconds")
    
    if results['performance']['cpu_times']:
        print("\nPerformance Summary (CPU):")
        for algo, time_taken in results['performance']['cpu_times'].items():
            print(f"  {algo}: {time_taken:.2f} seconds")