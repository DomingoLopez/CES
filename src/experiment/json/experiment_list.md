
| ID  | Dino Model | Optimizer | Optuna Trials | Normalization | Scaler   | Dim. Red | Clustering | Eval Method       | Penalty | Penalty Range | Cache |
|---- |-----------|-----------|---------------|--------------|----------|----------|------------|-------------------|---------|---------------|-------|
| 0   | small    | optuna    | 400           | ✅          | None     | umap     | hdbscan    | silhouette       | None    | None          | 1     |
| 1   | small    | optuna    | 400           | ✅          | standard | umap     | hdbscan    | silhouette       | None    | None          | 1     |
| 2   | base     | optuna    | 400           | ✅          | None     | umap     | hdbscan    | silhouette       | None    | None          | 1     |
| 3   | base     | optuna    | 400           | ✅          | standard | umap     | hdbscan    | silhouette       | None    | None          | 1     |
| 4   | base     | optuna    | 400           | ✅          | None     | umap     | hdbscan    | davies_bouldin   | None    | None          | 1     |
| 5   | base     | optuna    | 400           | ✅          | standard | umap     | hdbscan    | davies_bouldin   | None    | None          | 1     |
| 6   | base     | optuna    | 400           | ✅          | None     | umap     | hdbscan    | silhouette_noise | range   | [100, 600]    | 1     |
| 7   | base     | optuna    | 400           | ✅          | standard | umap     | hdbscan    | davies_noise     | range   | [100, 600]    | 1     |
