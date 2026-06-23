# Analyzing the Second-Hand Car Market: Trends, Patterns, and Key Drivers

Big Data Project -  Faculty of Mathematics and Computer Science, University of Bucharest (June 2025).

A full big-data pipeline — from raw scraped listings to GPU-accelerated dimensionality reduction, regression, and clustering — applied to the Romanian second-hand vehicle market.

## Overview

The dataset (~800K listings, more then 1.1 gb of txt) was collected from three major Romanian car marketplaces: **OLX**, **Publi24**, and **Autovit**. It includes detailed vehicle specs (brand, model, engine, mileage, year), market info (price, currency, seller type), legal/environmental data (emissions, pollution standard, owner history), and geographic/descriptive metadata.

The project covers the full pipeline:
1. **EDA & preprocessing** — cleaning, currency normalization, outlier removal, encoding, scaling.
2. **Dimensionality reduction** — PCA, t-SNE, UMAP, GPU-accelerated via RAPIDS cuML.
3. **Regression** — predicting price with Linear Regression, Gradient Boosting, and XGBoost.
4. **Clustering** — K-Means and DBSCAN to segment the market into interpretable vehicle tiers.

## Dataset

- **Source:** Original, ethically crawled from OLX, Publi24, Autovit (Romania).
- **Size:** up to ~765K rows depending on feature completeness.
- **Key features:** `manufacturing_year`, `engine_hp`, `engine_cc`, `km_no`, `price`, `doors_no`, `make`, `fuel_type`, `transmission_type`, `body`.
- **The dataset is unfortunately private and permission from the author can be granted upon request.**

### Preprocessing pipeline
- Missing-value imputation: median (numerical), most-frequent (categorical).
- Currency normalization: RON → EUR (÷5), other currencies dropped.
- Outlier removal via IQR filtering on `engine_hp`, `engine_cc`, `km_no`, `price`.
- One-hot encoding for categoricals, `StandardScaler` for numericals.

## Dimensionality Reduction

Three techniques were compared, all GPU-accelerated (NVIDIA A100/T4 via RAPIDS cuML on Google Colab):

| Algorithm | Compute time |
|---|---|
| PCA | 0.5s |
| t-SNE | 29.6s |
| UMAP | 53.8s |

- **PCA**: Component 1 (~61% variance) separates "premium/new" vs "budget/used" cars; Component 2 separates "heavy-duty/performance" vs "efficient/modern"; Component 3 is dominated by door count.
- **t-SNE**: Dimension 1 separates fuel type/transmission (diesel/manual vs petrol/automatic); Dimension 2 reflects engine displacement and door count. Produces tighter, more distinct clusters than PCA.
- **UMAP**: Reveals an interconnected market structure — Component 1 ties to fuel type/regional brand preference, Component 2 to body/transmission segmentation, Component 3 to engine performance tier.

## Regression — Price Prediction

Models: **Linear Regression**, **Gradient Boosting**, **XGBoost**, each trained on both PCA- and UMAP-reduced features (80/20 split), with `RandomizedSearchCV` hyperparameter tuning for the tree-based models.

| Model | Reduction | RMSE | R² |
|---|---|---|---|
| XGBoost | PCA | 41,496 | 0.6655 |
| XGBoost | UMAP | 26,257 | **0.8661** |
| Gradient Boosting | PCA | 9,394 | **0.9829** |
| Gradient Boosting | UMAP | 49,010 | 0.5334 |
| Linear Regression | PCA | 61,487 | 0.2657 |
| Linear Regression | UMAP | 71,742 | 0.0003 |

**Key finding:** the best dimensionality-reduction technique is model-dependent — UMAP paired best with XGBoost, PCA paired best with Gradient Boosting. Dimensionality reduction also cut XGBoost training time from ~11.1s (raw) to ~2–2.4s (PCA/UMAP).

## Clustering — Market Segmentation

**K-Means** and **DBSCAN** applied to a sampled, PCA-reduced subset; evaluated via Silhouette Score.

| Algorithm | Original data | PCA data |
|---|---|---|
| K-Means | 0.1856 | 0.3077 |
| DBSCAN | 0.2804 | 0.5602 |

- **K-Means (best k found)** produced 4 clusters: budget/average, older high-mileage station wagons, economical city cars (Dacia/Renault), and premium high-performance vehicles (BMW/Mercedes/Audi).
- **DBSCAN** isolated a balanced main cluster, a sporty/compact 2-door segment (BMW/Mercedes/Audi coupés), and a noise cluster of rare/niche makes (e.g. MG, Fiat) useful for anomaly/outlier detection.
- K-Means gave clearer, more interpretable market-tier segmentation; DBSCAN was better at surfacing niche segments and outliers without needing a predefined cluster count.

## Tech Stack

- **Data processing:** pandas, NumPy
- **GPU acceleration:** RAPIDS cuML, cuDF, CuPy (NVIDIA A100/T4 on Google Colab)
- **ML:** scikit-learn (PCA, t-SNE, K-Means, DBSCAN, Linear/Gradient Boosting Regression), XGBoost, UMAP
- **Visualization:** Matplotlib, Seaborn

## Repository Structure

```
.
├── Big Data Project.pdf                  # Full written report
├── src/
│   ├── BigDataProject.ipynb              # Main end-to-end notebook (EDA, dim. reduction, regression, clustering)
│   ├── AnalizareDate.py                   # EDA / data analysis script
│   ├── FeatureSelection+UMAP.py           # Feature selection + UMAP reduction
│   ├── ProcesareDate cu PCA,UMAP,tSNE.py  # PCA / UMAP / t-SNE processing pipeline
│   └── ReducereDimensiuniCuGpu.py         # GPU-accelerated dimensionality reduction (RAPIDS)
├── Graphs Dimensional Reduction/          # PCA/t-SNE/UMAP plots + component analysis (full dataset)
├── Graphs Dimensional Reduction on Sample dataset/  # Same, on sampled data
└── Sample Dataset/                       # Sample of the raw data
```

## Authors

Smarandi Anastasia, Gherghisan Sebastian Mihai, Orghidan Radu, Isfan Mihai George — Faculty of Mathematics and Computer Science, University of Bucharest.


# THIS IS A FINISHED PROJECT AND NO FURTHER COMMITS WILL BE MADE
