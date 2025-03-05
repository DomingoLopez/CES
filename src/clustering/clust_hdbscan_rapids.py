from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# RAPIDS: HDBSCAN GPU
from cuml.cluster import HDBSCAN

from sklearn.datasets import make_blobs

# Importar tu clase base
from src.clustering.clustering_model import ClusteringModel

# Para logging
from loguru import logger


class HDBSCANRapidsClustering(ClusteringModel):
    """
    HDBSCAN clustering model class using RAPIDS (GPU) inheriting from ClusteringModel.

    This class implements the HDBSCAN clustering algorithm from RAPIDS cuML on a dataset 
    and provides methods to run clustering, calculate metrics, and save results including 
    plots and score data.
    """

    def __init__(self, data: pd.DataFrame, **kwargs):
        """
        Initialize the HDBSCANRapidsClustering model.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be clustered.
        **kwargs : dict
            Additional parameters for customization or future expansion.
        """
        super().__init__(data, model_name="hdbscan_rapids")

    def run(self, **kwargs):
        """
        Run the HDBSCAN (RAPIDS) clustering with default or specified parameters.

        This method simply constructs an HDBSCAN model using default parameters 
        (or overridden by kwargs), fits it to the data, and stores the results 
        for further analysis or plotting.
        """
        logger.info("Running HDBSCAN with RAPIDS...")

        # Construimos el modelo con los parámetros que se pasen por kwargs
        # (o defaults si no se pasan).
        self.model = HDBSCAN(**kwargs)

        # Ajustamos el modelo
        # IMPORTANTE: la implementación de cuML maneja arrays tipo GPU
        # pero en muchos casos DataFrame de pandas se transforma internamente. 
        # Asegúrate de que sea float32 si tienes problemas de tipos.
        self.labels_ = self.model.fit_predict(self.data.values.astype(np.float32))
        
        # Guarda los labels en la clase base
        self.clusters_ = self.labels_

        # Llama, si lo deseas, a un método de la clase base para calcular
        # las métricas de clustering y guardarlas (p.e. silhouette).
        self.calculate_metrics()

        logger.info("HDBSCAN clustering with RAPIDS complete.")

    def run_optuna(self,
                   evaluation_method="silhouette",
                   n_trials=50,
                   penalty: str = "linear",
                   penalty_range: tuple = (2, 8)):
        """
        Run Optuna optimization for the HDBSCAN clustering model with a specified evaluation method.

        This method sets up and executes an Optuna hyperparameter optimization for the HDBSCAN 
        clustering algorithm from RAPIDS cuML. It defines the range of hyperparameters 
        specific to HDBSCAN, including `min_cluster_size`, `min_samples`, etc., and passes
        these parameters to the generic Optuna optimization method inherited from the base class.

        Parameters
        ----------
        evaluation_method : str, optional
            The evaluation metric to optimize. It can be either 'silhouette' (for maximizing 
            the silhouette score) or 'davies_bouldin' (for minimizing the Davies-Bouldin score). 
            Defaults to "silhouette".
        n_trials : int, optional
            The number of optimization trials to run. Defaults to 50.
        penalty : str, optional
            ...
        penalty_range : tuple, optional
            ...

        Returns
        -------
        optuna.study.Study
            The Optuna study object containing details of the optimization process, including 
            the best hyperparameters found and associated evaluation score.
        """
        def model_builder(trial):
            return HDBSCAN(
                # Ajusta los rangos a los que quieras explorar
                min_cluster_size=trial.suggest_int('min_cluster_size', 2, 8),
                min_samples=trial.suggest_int('min_samples', 2, 4),
                cluster_selection_epsilon=trial.suggest_float('cluster_selection_epsilon', 0.3, 3),
                metric=trial.suggest_categorical('metric', ['l2', 'euclidean']),
                cluster_selection_method=trial.suggest_categorical('cluster_selection_method', ['eom', 'leaf']),
                gen_min_span_tree=trial.suggest_categorical('gen_min_span_tree', [True, False])
            )

        # Asegúrate de que tu método run_optuna_generic sea compatible con 
        # la implementación cuML (que usa .fit_predict en lugar de .fit / .predict).
        return self.run_optuna_generic(model_builder, evaluation_method, n_trials,penalty, penalty_range)



if __name__ == "__main__":
    # Test de ejemplo
    data, _ = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=0.6,
        random_state=0
    )
    data = pd.DataFrame(data, columns=['x', 'y'])

    # Instanciamos la clase GPU
    hdbscan_rapids_clustering = HDBSCANRapidsClustering(data)

    # Ejemplo de ejecución con algunos parámetros
    hdbscan_rapids_clustering.run(min_cluster_size=5, min_samples=3)
    print("Labels:", hdbscan_rapids_clustering.labels_)

    # Si tus métodos genéricos (Optuna, GridSearch) están adaptados para cuML, 
    # podrías probar:
    # hdbscan_rapids_clustering.run_optuna()
    # hdbscan_rapids_clustering.run_gridsearch()
