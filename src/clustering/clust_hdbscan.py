from datetime import datetime
import hdbscan
from pathlib import Path
from typing import Optional
import pandas as pd
from loguru import logger
from sklearn.datasets import make_blobs
from src.clustering.clustering_model import ClusteringModel


class HDBSCANClustering(ClusteringModel):
    """
    HDBSCAN clustering model class inheriting from ClusteringModel.

    This class implements the HDBSCAN clustering algorithm on a dataset and 
    provides methods to run clustering, calculate metrics, and save results 
    including plots and score data.
    """

    def __init__(self, data: pd.DataFrame, **kwargs):
        """
        Initialize the HDBSCANClustering model.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be clustered.
        **kwargs : dict
            Additional parameters for customization or future expansion.
        """
        super().__init__(data, model_name="hdbscan")
    



    def run_optuna(self, evaluation_method="silhouette", n_trials=50,penalty: str="linear", penalty_range: tuple =(2,8)):
        """
        Run Optuna optimization for the HDBSCAN clustering model with a specified evaluation method.

        This method sets up and executes an Optuna hyperparameter optimization for the HDBSCAN 
        clustering algorithm. It defines the range of hyperparameters specific to HDBSCAN, 
        including `min_cluster_size`, `min_samples`, and `alpha`, and passes these parameters 
        to the generic Optuna optimization method inherited from the base class.
        
        Parameters
        ----------
        evaluation_method : str, optional
            The evaluation metric to optimize. It can be either 'silhouette' (for maximizing 
            the silhouette score) or 'davies_bouldin' (for minimizing the Davies-Bouldin score). 
            Defaults to "silhouette".
        n_trials : int, optional
            The number of optimization trials to run. Defaults to 50.

        Returns
        -------
        optuna.study.Study
            The Optuna study object containing details of the optimization process, including 
            the best hyperparameters found and associated evaluation score.
        
        Notes
        -----
        - This method calls the generic `run_optuna_generic` method from the base class, 
        which handles the Optuna optimization process and evaluation.
        - `model_builder` is a nested function that constructs an HDBSCAN model using 
        hyperparameters suggested by each Optuna trial.
        """
        # Param/model builder for hdbscan
        # EXPERIMENTOS 6,7,8,9,10,11
        def model_builder(trial):
            return hdbscan.HDBSCAN(
                min_cluster_size=trial.suggest_int('min_cluster_size', 2, 6),
                min_samples=trial.suggest_int('min_samples', 2, 4),
                cluster_selection_epsilon=trial.suggest_float('cluster_selection_epsilon', 0.0, 0.5),
                metric=trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev']),
                cluster_selection_method=trial.suggest_categorical('cluster_selection_method',['eom','leaf']),
                gen_min_span_tree=trial.suggest_categorical('gen_min_span_tree', [True, False]),
                # Para garantizar determinismo.
                approx_min_span_tree=False
            )
        # EXPERIMENTOS 0,2,3,4,5
        # def model_builder(trial):
        #     return hdbscan.HDBSCAN(
        #         min_cluster_size=trial.suggest_int('min_cluster_size', 2, 8),
        #         min_samples=trial.suggest_int('min_samples', 2, 4),
        #         cluster_selection_epsilon=trial.suggest_float('cluster_selection_epsilon', 0.3, 3),
        #         metric=trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev']),
        #         cluster_selection_method=trial.suggest_categorical('cluster_selection_method',['eom','leaf']),
        #         gen_min_span_tree=trial.suggest_categorical('gen_min_span_tree', [True, False]),
        #         # Para garantizar determinismo.
        #         approx_min_span_tree=False
        #     )
        # return hdbscan.HDBSCAN(
        #         min_cluster_size=trial.suggest_int('min_cluster_size', 2, 20),
        #         min_samples=trial.suggest_int('min_samples', 2, 20),
        #         cluster_selection_epsilon=trial.suggest_float('cluster_selection_epsilon', 0.01, 1.0, log=True),
        #         alpha=trial.suggest_float('alpha', 0.3, 1.5),
        #         metric=trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev']),
        #         #cluster_selection_method=trial.suggest_categorical('cluster_selection_method', ['eom', 'leaf']),
        #         cluster_selection_method=trial.suggest_categorical('cluster_selection_method',['eom','leaf']),
        #         gen_min_span_tree=trial.suggest_categorical('gen_min_span_tree', [True, False])
        #     )
        # Call generic class method
        return self.run_optuna_generic(model_builder, evaluation_method, n_trials,penalty, penalty_range)



if __name__ == "__main__":
    # Test the KMeansClustering class with a sample dataset
    data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
    data = pd.DataFrame(data, columns=['x', 'y'])
    kmeans_clustering = HDBSCANClustering(data)
    kmeans_clustering.run()
    print("HDBSCAN clustering complete. Results and plots saved.")
