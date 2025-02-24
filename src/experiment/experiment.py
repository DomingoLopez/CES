from collections import Counter
from itertools import product
import os
from pathlib import Path
import pickle
import tempfile
import sys
from typing import Optional
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.datasets import make_blobs
from src.clustering.clustering_factory import ClusteringFactory
from src.clustering.clustering_model import ClusteringModel
from src.preprocess.preprocess import Preprocess
import mlflow

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning
)


class Experiment():
    """
    Experiment Class where we can define which kind of methods, algorithms, 
    scalers and optimizers we should experiment with.
    
    This is the main class for setting up and running clustering experiments 
    using specified optimizers, dimensionality reduction methods, and evaluation metrics.
    """

    def __init__(self, 
                 id:int = 0,
                 bbdd:str = "test",
                 dino_model = "small",
                 data:pd.DataFrame = None, 
                 optimizer:str = "optuna",
                 optuna_trials: int = 100,
                 normalization:bool = True,
                 dim_red:str = None, 
                 reduction_params:dict = None,
                 scaler:str = None, 
                 clustering:str = "hdbscan",
                 eval_method:str = "silhouette",
                 penalty = None,
                 penalty_range = None,
                 cache= True, 
                 verbose= False,
                 **kwargs):
    
        """
        Initializes an experiment with the specified configuration.

        Args:
            data (list): The data to be used for the experiment.
            optimizer (str): The optimization method to use, e.g., 'optuna' or 'gridsearch'.
            dim_red (str): Dim reduction
            reduction_parameters (dict): parameters of reduction
            scalers (list): List of scalers to normalize the data.
            clustering (str): Clustering algorithm to apply.
            eval_method (str): Evaluation metric for clustering quality.
            penalty (str): Penalty type to be applied in optimization.
            penalty_range (tuple): Range of penalty values.
            cache (bool): If True, caching is enabled.
            verbose (bool): If True, enables verbose logging.
            **kwargs: Additional keyword arguments.
        """
        # Setup attrs
        self._id = id
        self._bbdd = bbdd
        self._dino_model = dino_model
        self._data = data
        self._optimizer = optimizer
        self._optuna_trials = optuna_trials
        self._normalization = normalization
        self._dim_red = dim_red
        self._reduction_params = reduction_params
        self._scaler = scaler
        self._clustering = clustering
        self._eval_method = eval_method
        self._penalty = penalty
        self._penalty_range = penalty_range
        self._cache = cache
        self._verbose = verbose
        self._results_df = None

        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        # Set experiment name
        self._experiment_name = f"{self._id}_{self._bbdd}_{self._dino_model}_{self._dim_red}_{self._eval_method}"


    # Getters and Setters
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    
    @property
    def bbdd(self):
        return self._bbdd

    @bbdd.setter
    def bbdd(self, value):
        self._bbdd = value

    @property
    def dino_model(self):
        return self._dino_model

    @dino_model.setter
    def dino_model(self, value):
        self._dino_model = value
        
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def optuna_trials(self):
        return self._optuna_trials

    @optuna_trials.setter
    def optuna_trials(self, value):
        self._optuna_trials = value

    @property
    def normalization(self):
        return self._normalization

    @normalization.setter
    def normalization(self, value):
        self._normalization = value

    @property
    def dim_red(self):
        return self._dim_red

    @dim_red.setter
    def dim_red(self, value):
        self._dim_red = value

    @property
    def reduction_params(self):
        return self._reduction_params

    @reduction_params.setter
    def reduction_params(self, value):
        self._reduction_params = value

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, value):
        self._scaler = value

    @property
    def clustering(self):
        return self._clustering

    @clustering.setter
    def clustering(self, value):
        self._clustering = value

    @property
    def eval_method(self):
        return self._eval_method

    @eval_method.setter
    def eval_method(self, value):
        self._eval_method = value

    @property
    def penalty(self):
        return self._penalty

    @penalty.setter
    def penalty(self, value):
        self._penalty = value

    @property
    def penalty_range(self):
        return self._penalty_range

    @penalty_range.setter
    def penalty_range(self, value):
        self._penalty_range = value

    @property
    def cache(self):
        return self._cache

    @cache.setter
    def cache(self, value):
        self._cache = value

    @property
    def results_df(self):
        return self._results_df

    @results_df.setter
    def results_df(self, value):
        self._results_df = value
    
       

    def run_experiment(self):
        """
        Executes the experiment based on the chosen optimizer.
        Calls the appropriate internal method for running an experiment using 
        Optuna optimizer.
        Raises:
            ValueError: If the optimizer specified is not supported.
        """
        logger.info(f"STARTING EXPERIMENT USING {self._optimizer.upper()} OPTIMIZER")
        if self._optimizer == "optuna":
            self.__run_experiment()
        else:
            raise ValueError("optimizer not supported. Valid options are 'optuna' or 'gridsearch' ")
    

    def __run_experiment(self):
        """
        Runs the experiment using the Optuna optimizer, performs optimization, 
        and saves results to mlflow.
        """
        # If cache, get experiment and end if it exists and is not deleted, else delete and execute
        experiment = self.__get_experiment_by_name(self._experiment_name)
        # En mlflow no podemos borrar para siempre a través de la api un experimento.
        if self._cache:
            if experiment is not None:
                return
            
        # Nombre del experimento
        mlflow.set_experiment(f"{self._id}_{self._bbdd}_{self._dino_model}_{self._dim_red}_{self._eval_method}")

        param_combinations = self.__get_param_combinations()

        for i,reduction_params in enumerate(param_combinations):
            # Trackeamos experiment
            with mlflow.start_run(run_name=f"run_{str(i)}"):
                embeddings = self.__apply_preprocessing(reduction_params)

                clustering_model = ClusteringFactory.create_clustering_model(self._clustering, embeddings)
                study = clustering_model.run_optuna(
                    evaluation_method=self._eval_method, n_trials=self._optuna_trials, penalty=self._penalty, penalty_range=self._penalty_range
                )
                best_trial = study.best_trial
                n_clusters_best = best_trial.user_attrs.get("n_clusters", None)
                centers_best = best_trial.user_attrs.get("centers", None)
                labels_best = best_trial.user_attrs.get("labels", None)
                label_counter = Counter(labels_best)
                score_best = best_trial.user_attrs.get("score_original", None)
                noise_not_noise = {
                    -1: label_counter.get(-1, 0),
                    1: sum(v for k, v in label_counter.items() if k != -1)
                }
                # Depending on eval_method type
                if self._eval_method == "silhouette":
                    score_noise_ratio = score_best / (noise_not_noise.get(-1) + 1)
                elif self._eval_method == "davies_bouldin":
                    score_noise_ratio = (noise_not_noise.get(-1) + 1) / score_best
                elif self._eval_method == "silhouette_noise":
                    score_noise_ratio = score_best
                elif self._eval_method == "davies_noise":
                    score_noise_ratio = score_best
                else:
                    raise ValueError(f"Unsupported evaluation method: {self._eval_method}")
                
                # Log de los parámetros
                mlflow.log_param("id", self._id)
                mlflow.log_param("bbdd", self._bbdd)
                mlflow.log_param("optimizer", self._optimizer)
                mlflow.log_param("clustering", self._clustering)
                mlflow.log_param("eval_method", self._eval_method)
                mlflow.log_param("optuna_trials", self._optuna_trials)
                mlflow.log_param("normalization", self._normalization)
                mlflow.log_param("scaler", self._scaler)
                mlflow.log_param("dim_red", self._dim_red)
                mlflow.log_param("reduction_params", reduction_params)
                mlflow.log_param("dimensions", reduction_params.get("n_components", None))
                mlflow.log_param("embeddings", embeddings)
                mlflow.log_param("n_clusters", n_clusters_best)
                mlflow.log_param("best_params", str(study.best_params))
                mlflow.log_param("centers", centers_best)
                mlflow.log_param("labels", labels_best)
                mlflow.log_param("label_counter", label_counter)
                mlflow.log_param("noise_not_noise", noise_not_noise)
                mlflow.log_param("score_noise_ratio", score_noise_ratio)
                mlflow.log_param("penalty", self._penalty)
                mlflow.log_param("penalty_range", self._penalty_range)
                mlflow.log_metric("score_w_penalty", study.best_value)
                mlflow.log_metric("score_wo_penalty", score_best)
                logger.info("EXPERIMENT ENDED.")



    def __get_experiment_by_name(self,name):
        return mlflow.get_experiment_by_name(name)


    def __get_param_combinations(self):
        """
        Generates parameter combinations based on dimensionality reduction and reduction parameters.
        """
        if self._dim_red and self._reduction_params:
            param_names = list(self._reduction_params.keys())
            param_values = list(self._reduction_params.values())
            return [dict(zip(param_names, combination)) for combination in product(*param_values)]
        return [{}]


    def __apply_preprocessing(self, reduction_params):
        """
        Applies preprocessing steps including normalization, scaling, and dimensionality reduction.
        """
        preprocces_obj = Preprocess(embeddings=self._data, 
                                    bbdd=self._bbdd,
                                    dino_model = self._dino_model,
                                    scaler=self._scaler, 
                                    normalization=self._normalization,
                                    dim_red=self._dim_red,
                                    reduction_params=reduction_params)
        return preprocces_obj.run_preprocess()

    


if __name__ == "__main__":
    pass
