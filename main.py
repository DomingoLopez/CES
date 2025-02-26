import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json
from loguru import logger

from src.clustering.clust_hdbscan import HDBSCANClustering
from src.clustering.clustering_factory import ClusteringFactory
from src.experiment.experiment import Experiment
from src.experiment.experiment_result_controller import ExperimentResultController
from src.llava_inference.llava_inference import LlavaInference
from src.multimodal_clustering_metric.multimodal_clustering_metric import MultiModalClusteringMetric
from src.utils.image_loader import ImageLoader
from src.dinov2_inference.dinov2_inference import Dinov2Inference
from src.preprocess.preprocess import Preprocess

import matplotlib.pyplot as plt
import cv2


def load_images(path) -> list:
    # Finding images
    # image_loader = ImageLoader(folder="./data/Small_Data")
    image_loader = ImageLoader(folder=path)
    images = image_loader.find_images()
    return images

def generate_embeddings(images, model) -> list:
    # Loading images and getting embeddings
    dinomodel = Dinov2Inference(model_name=model, images=images, cache=True)
    embeddings = dinomodel.run()
    return embeddings


def run_experiments(file, images, bbdd) -> None:
   
    # Load json file with all experiments
    with open(file, 'r') as f:
        experiments_config = json.load(f)
    # For every configuration, run experiments
    for config in experiments_config:
        id = config.get("id")
        dino_model = config.get("dino_model","small")
        optimizer = config.get("optimizer", "optuna")
        optuna_trials = config.get("optuna_trials", None)
        normalization = config.get("normalization", True)
        scaler = config.get("scaler", None)
        dim_red = config.get("dim_red", None)
        reduction_parameters = config.get("reduction_parameters", None)
        clustering = config.get("clustering", "hdbscan")
        eval_method = config.get("eval_method", "silhouette")
        penalty = config.get("penalty", None)
        penalty_range = config.get("penalty_range", None)
        cache = config.get("cache", True)
        experiment_name = f"{id}_{bbdd}_{dino_model}_{dim_red}_{eval_method}"
        # Make and Run Experiment
        logger.info(f"LOADING EXPERIMENT: {id}")

        # Generate embeddings based on experiment model
        embeddings = generate_embeddings(images, model=dino_model)
        experiment = Experiment(
            id,
            experiment_name,
            bbdd,
            dino_model,
            embeddings,
            optimizer,
            optuna_trials,
            normalization,
            dim_red,
            reduction_parameters,
            scaler,
            clustering,
            eval_method,
            penalty,
            penalty_range,
            cache
        )
        experiment.run_experiment()

        # Generate artifacts and results for every experiment. 
        experiment_controller = ExperimentResultController(eval_method=eval_method, 
                                                            n_images=len(images),
                                                            dino_model=dino_model,
                                                            dim_red=dim_red,
                                                            reduction_params=None,
                                                            n_cluster_range=None,
                                                            experiment_name=experiment_name)
        best_run = experiment_controller.get_top_k_runs(top_k=3)

        experiment_controller.create_cluster_dirs(images=images, runs=best_run, knn=None, copy_images=True )
        experiment_controller.create_plots(runs=best_run)


if __name__ == "__main__": 
    
    # ###################################################################
    # Path de test
    # image_path = "./data/test"
    # bbdd = "test"
    # experiments_file = "src/experiment/json/single.json"

    # 1. OBTAIN IMAGES
    image_path ="./data/flickr/flickr_validated_imgs_7000"
    bbdd = "flickr"
    experiments_file = "src/experiment/json/experiments_optuna_all.json"
    #experiments_file = "src/experiment/json/single.json"
    images = load_images(image_path)    


    # START EXPERIMENTS
    classification_lvl = [3]
    prompts = [2]
    n_lvlm_categories = 1
    llava_models = ["llava1-6_7b"]
    # Obtain experiments config
    with open(experiments_file, 'r') as f:
        experiments_config = json.load(f)

    result_list_top_trials = []
    # All experiment
    for config in experiments_config:
        id = config.get("id",1)
        dino_model = config.get("dino_model","small")
        optimizer = config.get("optimizer", "optuna")
        optuna_trials = config.get("optuna_trials", None)
        normalization = config.get("normalization", True)
        scaler = config.get("scaler", None)
        dim_red = config.get("dim_red", "umap")
        reduction_parameters = config.get("reduction_parameters", None)
        clustering = config.get("clustering", "hdbscan")
        eval_method = config.get("eval_method", "silhouette")
        penalty = config.get("penalty", None)
        penalty_range = config.get("penalty_range", None)
        cache = config.get("cache", True)
        experiment_name = f"{id}_{bbdd}_{dino_model}_{dim_red}_{eval_method}"
        logger.info(f"LOADING EXPERIMENT: {id}")

        # 2. GENERATE EMBEDDINGS FOR EACH EXPERIMENT
        embeddings = generate_embeddings(images, model=dino_model)
        experiment = Experiment(
            id,
            experiment_name,
            bbdd,
            dino_model,
            embeddings,
            optimizer,
            optuna_trials,
            normalization,
            dim_red,
            reduction_parameters,
            scaler,
            clustering,
            eval_method,
            penalty,
            penalty_range,
            cache
        )
        # 3. APPLY CLUSTERING AND GENERATE PLOTS
        experiment.run_experiment()
        # Generate artifacts and results for every experiment. 
        experiment_controller = ExperimentResultController(eval_method=eval_method, 
                                                            n_images=len(images),
                                                            dino_model=dino_model,
                                                            dim_red=dim_red,
                                                            reduction_params=None,
                                                            n_cluster_range=None,
                                                            experiment_name=experiment_name)
        best_runs = experiment_controller.get_top_k_runs(top_k=3)
        experiment_controller.create_cluster_dirs(images=images, runs=best_runs, knn=None, copy_images=False)
        experiment_controller.create_plots(runs=best_runs)

    #     # 4. RUN LLAVA INFERENCE
    #     for class_lvl in classification_lvl:
    #         for model in llava_models:
    #             for prompt in prompts:
    #                 llava = LlavaInference(images=images, bbdd=bbdd, classification_lvl=class_lvl, n_prompt=prompt, model=model)
    #                 llava.run()
    #                 # Get Llava Results
    #                 llava_results_df = llava.get_results()
    #                 # Obtain categories
    #                 categories = llava.get_categories()

    #                 # 5. CALCULATE QUALITY METRICS
    #                 for idx, run in best_runs.iterrows():
    #                     img_cluster_dict = experiment_controller.get_cluster_images_dict(images,run,None,False)
    #                     # Quality metrics
    #                     lvm_lvlm_metric = MultiModalClusteringMetric(experiment_name,
    #                                                                 class_lvl,
    #                                                                 categories,
    #                                                                 model, 
    #                                                                 prompt, 
    #                                                                 run, 
    #                                                                 img_cluster_dict, 
    #                                                                 llava_results_df)
                        
    #                     if prompt != 3:
    #                         lvm_lvlm_metric.generate_stats()
    #                     elif prompt == 3 and n_lvlm_categories != 0:
    #                         lvm_lvlm_metric.generate_stats_multiple_categories(n_lvlm_categories)
    #                     else:
    #                         lvm_lvlm_metric.generate_stats()
                        
                        
    #                     # Obtain results
    #                     quality_results = pd.DataFrame()
    #                     for i in (True, False):
    #                         # Calculate metrics
    #                         results = lvm_lvlm_metric.calculate_clustering_quality(use_noise=i)
    #                         # Join results (in columns)
    #                         quality_results = pd.concat([quality_results, pd.DataFrame([results])], axis=1)


                        
    #                     # Save results in list
    #                     result_list_top_trials.append({
    #                         "experiment_id" : id,
    #                         "run_id": run["run_id"],
    #                         "dino_model" : dino_model,
    #                         "normalization" : run["params.normalization"],
    #                         "scaler" : run["params.scaler"],
    #                         "dim_red" : run["params.dim_red"],
    #                         "reduction_parameters" : run["artifacts.reduction_params"],
    #                         "clustering" : run["params.clustering"],
    #                         "n_clusters": run["params.n_clusters"],
    #                         "best_params": run["artifacts.best_params"],
    #                         "penalty" : run["params.penalty"],
    #                         "penalty_range" : run["params.penalty_range"],
    #                         "noise_not_noise" : run["artifacts.noise_not_noise"],
    #                         # Important things
    #                         "classification_lvl": class_lvl,
    #                         "lvlm": model,
    #                         "prompt": prompt,
    #                         "eval_method": eval_method,
    #                         "best_score": run["metrics.score_w_penalty"] if "noise" in run["params.eval_method"] else run["metrics.score_wo_penalty"], 
    #                         # Metrics
    #                         "homogeneity_global": quality_results["homogeneity_global"].iloc[0],
    #                         "entropy_global": quality_results["entropy_global"].iloc[0],
    #                         "quality_metric":quality_results["quality_metric"].iloc[0]
    #                         # "homogeneity_global_w_noise": quality_results["homogeneity_global_w_noise"].iloc[0],
    #                         # "entropy_global_w_noise": quality_results["entropy_global_w_noise"].iloc[0],
    #                         # "quality_metric_w_noise":quality_results["quality_metric_w_noise"].iloc[0]
    #                     })


    #                     lvm_lvlm_metric.plot_cluster_categories_3()


    # # df_results = pd.DataFrame(result_list)
    # # df_results.to_csv("results.csv",sep=";")

    # df_results_top_k = pd.DataFrame(result_list_top_trials)
    # df_results_top_k.to_csv("results_top_trials.csv",sep=";")
