import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import umap
import warnings
import time
import gc
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path, target_col='price', sample_size=None):
    """
    Load and preprocess the vehicle dataset, handling missing values and categorical features.
    
    Args:
        file_path: Path to the dataset file
        target_col: Target column for prediction
        sample_size: Number of rows to sample (None for all)
        
    Returns:
        Preprocessed X and y, plus column information
    """
    print(f"Loading data from {file_path}...")
    
    # Load data with appropriate settings for this format
    # The dataset has quoted fields, comma separation, and \N for NULL values
    df = pd.read_csv(file_path, na_values=[r'\N', 'missing', 'None', 'NONE', ''], 
                     low_memory=False, quotechar='"')
    
    # Sample the data if needed
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    print(f"Loaded DataFrame with shape: {df.shape}")
    
    # Identify the target column or find an alternative
    if target_col not in df.columns:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            target_col = numeric_cols[0]
            print(f"Target column not found. Using '{target_col}' as target instead")
    
    # Convert columns to numeric where possible
    for col in df.columns:
        if col != target_col:  # Don't convert target yet
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    # Convert target to numeric as well
    try:
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    except:
        print(f"Warning: Could not convert target column '{target_col}' to numeric")
    
    # Remove rows where target is null
    df = df.dropna(subset=[target_col])
    
    # Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []
    
    for col in df.columns:
        if col == target_col:
            continue
        
        # Check if column has few unique values relative to its size
        unique_ratio = df[col].nunique() / len(df)
        
        if df[col].dtype == 'object' or unique_ratio < 0.05:
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
    
    print(f"Found {len(numerical_cols)} numerical features and {len(categorical_cols)} categorical features")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y, numerical_cols, categorical_cols

def feature_importance_with_random_forest(X, y, numerical_cols, categorical_cols):
    """
    Use Random Forest to calculate feature importance scores.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        
    Returns:
        DataFrame with feature importance scores
    """
    print("\nApplying Random Forest Feature Selection...")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Create and fit the random forest model with the preprocessor
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Fit the model
    model.fit(X, y)
    
    # Get feature importances
    feature_names = numerical_cols.copy()
    
    # Add one-hot encoded categorical feature names if available
    if categorical_cols:
        try:
            cat_features = model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_features)
        except:
            # If something goes wrong with getting categorical feature names, use placeholders
            for col in categorical_cols:
                feature_names.append(f"{col}_encoded")
    
    # Extract importance scores
    importances = model.named_steps['model'].feature_importances_
    
    # If the lengths don't match, trim to the shorter length
    if len(importances) < len(feature_names):
        feature_names = feature_names[:len(importances)]
    elif len(importances) > len(feature_names):
        importances = importances[:len(feature_names)]
    
    # Create a DataFrame of feature importances
    feature_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importances

def mutual_information_feature_selection(X, y, numerical_cols, categorical_cols, k=20):
    """
    Apply Mutual Information feature selection.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        k: Number of features to select
        
    Returns:
        DataFrame with feature scores
    """
    print("\nApplying Mutual Information Feature Selection...")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Apply mutual information
    selector = SelectKBest(mutual_info_regression, k=min(k, X_processed.shape[1]))
    selector.fit(X_processed, y)
    
    # Get feature names
    feature_names = numerical_cols.copy()
    
    # Add one-hot encoded categorical feature names if available
    if categorical_cols:
        try:
            cat_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_features)
        except:
            # If something goes wrong with getting categorical feature names, use placeholders
            for col in categorical_cols:
                feature_names.append(f"{col}_encoded")
    
    # If the lengths don't match, trim to the shorter length
    scores = selector.scores_  # Initialize scores variable first
    
    if len(selector.scores_) < len(feature_names):
        feature_names = feature_names[:len(selector.scores_)]
    elif len(selector.scores_) > len(feature_names):
        scores = selector.scores_[:len(feature_names)]
    
    # Create a DataFrame of feature scores
    feature_scores = pd.DataFrame({
        'feature': feature_names,
        'score': scores
    }).sort_values('score', ascending=False)
    
    return feature_scores
def umap_dimensionality_reduction(X, y, numerical_cols, categorical_cols, n_components=2):
    """
    Apply UMAP for dimensionality reduction.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        n_components: Number of components for UMAP
        
    Returns:
        UMAP embedding and the preprocessed data
    """
    print("\nApplying UMAP Dimensionality Reduction...")
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(X_processed)
    
    return embedding, X_processed, reducer

def visualize_results(rf_importances, mi_scores, umap_embedding, y, output_prefix=""):
    """
    Create visualizations for the dimensionality reduction results.
    
    Args:
        rf_importances: Random Forest feature importances
        mi_scores: Mutual Information feature scores
        umap_embedding: UMAP embedding
        y: Target values
        output_prefix: Prefix for output filenames
    """
    print("\nCreating visualizations...")
    
    # 1. Random Forest Feature Importance
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=rf_importances.head(20))
    plt.title('Top 20 Features by Random Forest Importance')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}random_forest_features.png')
    plt.close()
    
    # 2. Mutual Information Scores
    plt.figure(figsize=(12, 10))
    sns.barplot(x='score', y='feature', data=mi_scores.head(20))
    plt.title('Top 20 Features by Mutual Information')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}mutual_info_features.png')
    plt.close()
    
    # 3. UMAP Visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                         c=y, cmap='viridis', alpha=0.5, s=5)
    plt.colorbar(scatter, label='Target Value')
    plt.title('UMAP Projection of the Dataset')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}umap_projection.png')
    plt.close()
    
    print("Visualizations saved as PNG files.")

def analyze_umap_dimensions(umap_model, X_processed, feature_names):
    """
    Analyze which original features contribute most to each UMAP dimension.
    
    Args:
        umap_model: Fitted UMAP model
        X_processed: Preprocessed feature data
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature contributions to UMAP dimensions
    """
    print("\nAnalyzing UMAP dimensions...")
    
    # Create generic feature names if the lengths don't match
    if len(feature_names) < X_processed.shape[1]:
        print(f"Warning: Number of feature names ({len(feature_names)}) doesn't match number of features "
              f"({X_processed.shape[1]}). Using generic feature names.")
        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
    
    # Create a DataFrame with UMAP dimensions
    umap_embedding = umap_model.transform(X_processed)
    embedding_df = pd.DataFrame(umap_embedding, columns=[f'UMAP{i+1}' for i in range(umap_embedding.shape[1])])
    
    # Create a DataFrame with preprocessed features
    features_df = pd.DataFrame(X_processed, columns=feature_names[:X_processed.shape[1]])
    
    # Calculate correlations
    correlations = {}
    for umap_dim in embedding_df.columns:
        # Calculate correlation between each feature and this UMAP dimension
        corr = features_df.corrwith(embedding_df[umap_dim]).abs()
        # Sort and get top contributors
        top_contributors = corr.sort_values(ascending=False).head(10)
        correlations[umap_dim] = top_contributors
    
    # Convert to DataFrame for easier reporting
    results = pd.DataFrame(correlations)
    
    return results

def main():
    """Main function to run the dimensionality reduction analysis."""
    start_time = time.time()
    print("Starting dimensionality reduction analysis...")
    
    # Load and preprocess data
    # For testing, we'll use a sample of the data
    X, y, numerical_cols, categorical_cols = load_and_preprocess_data('market_data.csv', sample_size=10000)
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply feature selection techniques
    # 1. Random Forest Feature Importance
    rf_importances = feature_importance_with_random_forest(X_train, y_train, numerical_cols, categorical_cols)
    
    # 2. Mutual Information Feature Selection
    mi_scores = mutual_information_feature_selection(X_train, y_train, numerical_cols, categorical_cols)
    
    # 3. UMAP Dimensionality Reduction
    umap_embedding, X_processed, umap_model = umap_dimensionality_reduction(X_train, y_train, numerical_cols, categorical_cols)
    
    # Get feature names for preprocessed data
    feature_names = numerical_cols.copy()
    if categorical_cols:
        try:
            # This assumes we're using the same preprocessor as in previous functions
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ]), numerical_cols),
                    ('cat', Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='most_frequent')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    ]), categorical_cols)
                ])
            preprocessor.fit(X_train)
            cat_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_features)
        except:
            # If something goes wrong, use placeholders
            for col in categorical_cols:
                feature_names.append(f"{col}_encoded")
    
    # Analyze UMAP dimensions
    umap_analysis = analyze_umap_dimensions(umap_model, X_processed, feature_names)
    
    # Create visualizations
    visualize_results(rf_importances, mi_scores, umap_embedding, y_train)
    
    # Print summary of results
    print("\nSummary of dimensionality reduction results:")
    print("\nTop 10 features by Random Forest Importance:")
    print(rf_importances.head(10).to_string(index=False))
    
    print("\nTop 10 features by Mutual Information:")
    print(mi_scores.head(10).to_string(index=False))
    
    print("\nTop contributors to UMAP dimensions:")
    print(umap_analysis.head(10))
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"\nDimensionality reduction analysis completed in {execution_time:.2f} seconds.")
    
    # Memory cleanup
    gc.collect()

if __name__ == "__main__":
    main()