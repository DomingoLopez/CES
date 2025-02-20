| ID | Dino Model | Normalization | Scaler   | Clustering | Dim. Reduction | Eval Method                    |
|----|-----------|---------------|----------|-----------|----------------|--------------------------------|
| 1  | small     | Sí            | –        | hdbscan   | umap           | silhouette                     |
| 2  | small     | Sí            | standard | hdbscan   | umap           | silhouette                     |
| 3  | base      | Sí            | –        | hdbscan   | umap           | silhouette                     |
| 4  | base      | Sí            | standard | hdbscan   | umap           | silhouette                     |
| 5  | base      | Sí            | –        | hdbscan   | umap           | davies_bouldin                 |
| 6  | base      | Sí            | standard | hdbscan   | umap           | davies_bouldin                 |
| 7  | base      | Sí            | –        | hdbscan   | umap           | silhouette_noise (100–600)     |
| 8  | base      | Sí            | standard | hdbscan   | umap           | davies_noise (100–600)         |
