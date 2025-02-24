from collections import Counter
from itertools import product
import json
import os
import ast
from pathlib import Path
import pickle
import shutil
import sys
from matplotlib.colors import ListedColormap
import mlflow
import mlflow.entities
import mlflow.tracking
import seaborn as sns
from typing import NewType, Optional
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import umap
from src.clustering.clustering_factory import ClusteringFactory
from src.clustering.clustering_model import ClusteringModel
from src.preprocess.preprocess import Preprocess


class ExperimentResultController():

    def __init__(self, 
                 eval_method="silhouette",
                 n_images=7072,
                 dino_model="small",
                 dim_red="umap",
                 experiment_name="1",
                 n_cluster_range = None,
                 reduction_params = None,
                 cache= True, 
                 verbose= False,
                 **kwargs):
    
        # Setup attrs
        self.eval_method = eval_method
        self.n_images = n_images
        self.dino_model = dino_model
        self.experiment_name = experiment_name
        self.cache = cache
        self.verbose = verbose
        
        # Filters for different configurations
        self.n_cluster_range  = (100,500) if n_cluster_range is None else n_cluster_range
        if reduction_params is None:
            if dim_red == "umap":
                self.reduction_params = {
                    "n_components": (2,25),
                    "n_neighbors": (3,60),
                    "min_dist": (0.1, 0.8)
                }
            elif dim_red == "tsne":
                self.reduction_params = {
                    "n_components": (2,25),
                    "perplexity": (4,60),
                    "early_exaggeration": (7, 16)
                }
            else:
                self.reduction_params = {
                    "n_components": (2,25)
                }
        else:
            self.reduction_params = reduction_params


        logger.remove()
        if verbose:
            logger.add(sys.stdout, level="DEBUG")
        else:
            logger.add(sys.stdout, level="INFO")

        # Clusters dir
        self.cluster_dir = (
            Path(__file__).resolve().parent
            / f"clusters"
        )
        self.cluster_dir.mkdir(parents=True, exist_ok=True)

        # # Load all runs for given experiment
        # self.results_df = None
        # self.cluster_images_dict = None
        # self.__load_all_runs()

        # Get original embeddings (Just for representation)
        # This is a bit messy. Refactor asap.
        if self.dino_model == "small":
            embeddings_name = f"embeddings_dinov2_vits14_{self.n_images}.pkl"
        else:
            embeddings_name = f"embeddings_dinov2_vitb14_{self.n_images}.pkl"
        original_embeddings_path = Path(__file__).resolve().parent.parent / f"dinov2_inference/cache/{embeddings_name}"
        with open(original_embeddings_path, "rb") as f:
            self.original_embeddings = pickle.load(f)
        
        # Para descargar artefactos desde mlflow
        self.mlflowclient = mlflow.tracking.MlflowClient()



    def apply_filters_and_obtain_top_k_best(self, df, top_k=1):
 
        # Validate n_cluster_range
        min_n_cluster, max_n_cluster = self.n_cluster_range
        if min_n_cluster < 2 or max_n_cluster > 800:
            raise ValueError("n_cluster_range values must be between 2 and 800.")
        if min_n_cluster > max_n_cluster:
            raise ValueError("min_n_cluster cannot be greater than max_n_cluster.")
        
        # Validate reduction_params
        for key, value_range in self.reduction_params.items():
            if not isinstance(value_range, tuple) or len(value_range) != 2:
                raise ValueError(f"Parameter {key} in reduction_params must be a tuple (min, max).")
            if value_range[0] > value_range[1]:
                raise ValueError(f"Invalid range for {key}: {value_range}. Min cannot be greater than Max.")
    
        
        # Verify df is loaded
        if df.shape[0] == 0 or df is None:
            logger.warning("No runs loaded. Returning an empty DataFrame.")
            return pd.DataFrame()


        # Determine sorting column and order based on eval_method
        if self.eval_method == "davies_bouldin":
            sort_column =  'metrics.score_wo_penalty'
            ascending_order = True  # Lower is better for davies_bouldin
        elif self.eval_method == "davies_noise":
            sort_column = 'metrics.score_w_penalty'
            ascending_order = True  # Lower is better for davies_bouldin
        elif self.eval_method == "silhouette":
            sort_column =  'metrics.score_wo_penalty'
            ascending_order = False  # Higher is better for silhouette
        elif self.eval_method == "silhouette_noise":
            sort_column =  'metrics.score_w_penalty'
            ascending_order = False  # Higher is better for silhouette
        else:
            raise ValueError("Eval method not supported")


        # Filter dataframe based on cluster
        filtered_df = df[
            (df['params.n_clusters'] >= min_n_cluster) & 
            (df['params.n_clusters'] <= max_n_cluster) 
        ]

        # Check if df empty and filter based on reduction params
        if not filtered_df.empty:
            # Filter by reduction params
            for param, value_range in self.reduction_params.items():
                min_val, max_val = value_range
                filtered_df = filtered_df[
                    filtered_df['artifacts.reduction_params'].apply(
                        lambda params: param in params and min_val <= params[param] <= max_val
                    )
                ]
        else:
             logger.warning("Column 'reduction_params' not found or DataFrame is empty. Skipping reduction parameter filtering.")


        if not filtered_df.empty:
            filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending_order)
        else:
            logger.warning("Filtered DataFrame is empty. Skipping sorting step.")


        if filtered_df.empty:
            logger.warning("Filtered DataFrame is empty after applying filters. Returning top_k runs from the entire dataset.")
            filtered_df = df.sort_values(by=sort_column, ascending=ascending_order)

        # Select the top_k runs
        top_k_df = filtered_df.head(top_k)
        
        return top_k_df




    def add_artifacts_data(self, df):
        """
        Helper function to add artifact data to a DataFrame.

        Parameters
        ----------
        df : DataFrame
            The DataFrame containing MLflow runs.

        Returns
        -------
        pd.DataFrame
            The DataFrame with artifacts added as new columns.
        """ 
        
        # Inicializar listas para los artifacts
        best_params_list = []
        centers_list = []
        labels_list = []
        embeddings_list = []
        label_counter_list = []
        noise_not_noise_list = []
        reduction_params_list = []

        for _, row in df.iterrows():  # Iterar sobre cada fila del DataFrame
            run_id = row["run_id"]  # Obtener el run_id de la fila actual

            try:
                # Descargar artifacts
                localpath_best_params = self.mlflowclient.download_artifacts(run_id, "best_params.json")
                localpath_labels = self.mlflowclient.download_artifacts(run_id, "labels.pkl")
                localpath_centers = self.mlflowclient.download_artifacts(run_id, "centers.pkl")
                localpath_embeddings = self.mlflowclient.download_artifacts(run_id, "embeddings.pkl")
                localpath_label_counter = self.mlflowclient.download_artifacts(run_id, "label_counter.json")
                localpath_noise_not_noise = self.mlflowclient.download_artifacts(run_id, "noise_not_noise.json")
                localpath_reduction_params = self.mlflowclient.download_artifacts(run_id, "reduction_params.json")

                # Cargar archivos JSON
                with open(localpath_best_params, "r") as f:
                    best_params = json.load(f)
                with open(localpath_label_counter, "r") as f:
                    label_counter = json.load(f)
                with open(localpath_noise_not_noise, "r") as f:
                    noise_not_noise = json.load(f)
                with open(localpath_reduction_params, "r") as f:
                    reduction_params = json.load(f)

                # Cargar archivos Pickle
                with open(localpath_centers, "rb") as f:
                    centers = pickle.load(f)
                with open(localpath_embeddings, "rb") as f:
                    embeddings = pickle.load(f)
                with open(localpath_labels, "rb") as f:
                    labels = pickle.load(f)

            except Exception as e:
                logger.warning(f"Could not load artifacts for run_id={run_id}: {e}")
                best_params = None
                labels = None
                centers = None
                embeddings = None
                label_counter = None
                noise_not_noise = None
                reduction_params = None

            # Agregar artifacts a las listas
            best_params_list.append(best_params)
            centers_list.append(centers)
            labels_list.append(labels)
            embeddings_list.append(embeddings)
            label_counter_list.append(label_counter)
            noise_not_noise_list.append(noise_not_noise)
            reduction_params_list.append(reduction_params)

        # Agregar las nuevas columnas al DataFrame
        df["artifacts.best_params"] = best_params_list
        df["artifacts.labels"] = labels_list
        df["artifacts.label_counter"] = label_counter_list
        df["artifacts.noise_not_noise"] = noise_not_noise_list
        df["artifacts.reduction_params"] = reduction_params_list
        df["artifacts.centers"] = centers_list
        df["artifacts.embeddings"] = embeddings_list

        return df



    def get_top_k_runs(self,top_k):

        try:
            # 1) Get the experiment_id from its name
            if self.experiment_name:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    raise ValueError(f"Experiment '{self.experiment_name}' was not found in MLflow.")
                experiment_ids = [experiment.experiment_id]
            else:
                raise ValueError("No experiment_name was provided.")


            # 2) Retrieve MLflow runs (finished ones, for instance)
            runs_df = mlflow.search_runs(
                experiment_ids=experiment_ids,
                run_view_type=mlflow.entities.ViewType.ALL,  # Include all runs, not only active ones
                filter_string='attribute.status = "FINISHED"'  # Only include finished runs
            )

            #Transforms some columns
            runs_df["params.dimensions"] = pd.to_numeric(runs_df['params.dimensions'], errors='coerce')
            runs_df["params.n_clusters"] = pd.to_numeric(runs_df['params.n_clusters'], errors='coerce')


            # Filter
            filtered_df = self.apply_filters_and_obtain_top_k_best(runs_df,top_k = top_k)
            df = self.add_artifacts_data(filtered_df)
            return df

        except Exception as e:
            # In case of any error, leave an empty DataFrame and log a warning
            logger.warning(f"Could not load runs from MLflow for experiment_name='{self.experiment_name}': {e}")




    def get_cluster_images_dict(self, images, run, knn=None, save_result=True):
        """
        Finds the k-nearest neighbors for each centroid of clusters among points that belong to the same cluster.
        Returns knn points for each cluster in dict format in case knn is not None

        Parameters
        ----------
        knn : int
            Number of nearest neighbors to find for each centroid

        Returns
        -------
        sorted_cluster_images_dict : dictionary with images per cluster (as key)
        """

        cluster_images_dict = {}
        labels = run['artifacts.labels']

        if knn is not None:
            used_metric = "euclidean"
            
            for idx, centroid in enumerate(tqdm(run['artifacts.centers'], desc="Processing cluster dirs (knn images selected)")):
                # Filter points based on label mask over embeddings
                cluster_points = run['artifacts.embeddings'].values[labels == idx]
                cluster_images = [images[i] for i in range(len(images)) if labels[i] == idx]
                # Adjust neighbors, just in case
                n_neighbors_cluster = min(knn, len(cluster_points))
                
                nbrs = NearestNeighbors(n_neighbors=n_neighbors_cluster, metric=used_metric, algorithm='auto').fit(cluster_points)
                distances, indices = nbrs.kneighbors([centroid])
                closest_indices = indices.flatten()
                
                # Get images for each cluster
                cluster_images_dict[idx] = [cluster_images[i] for i in closest_indices]

            # Get noise (-1)
            cluster_images_dict[-1] = [images[i] for i in range(len(images)) if labels[i] == -1]
            
        else:
            for i, label in enumerate(tqdm(labels, desc="Processing cluster dirs")):
                if label not in cluster_images_dict:
                    cluster_images_dict[label] = []
                cluster_images_dict[label].append(images[i])
        
        # Sort dictionary
        if save_result:
            self.cluster_images_dict = dict(sorted(cluster_images_dict.items()))
        return self.cluster_images_dict




    def get_cluster_run_path(self, run):
        return os.path.join(self.cluster_dir, f"{self.experiment_name}/run_{run['run_id']}_{self.eval_method}_{run['metrics.score_wo_penalty']:.3f}")




    def create_cluster_dirs(self, images, runs, knn=None, copy_images=True):
        """
        Create a dir for every cluster given in dictionary of images. 
        This is how we are gonna send that folder to ugr gpus
        """
        # logger.info("Copying images from Data path to cluster dirs")
        # For every key (cluster index)
        
        for i,r in runs.iterrows():
            images_dict_format = self.get_cluster_images_dict(images, r)
            path_cluster = os.path.join(self.get_cluster_run_path(r), "clusters")
            cluster_data = []
            try:
                for cluster_id, image_paths in images_dict_format.items():
                    # Create folder if it doesnt exists
                    cluster_dir = os.path.join(path_cluster, str(cluster_id)) 
                    os.makedirs(cluster_dir, exist_ok=True)
                    # For every path image, copy that image from its path to cluster folder
                    for image_path in image_paths:
                        cluster_data.append([cluster_id, image_path])
                        if copy_images:
                            shutil.copy(image_path, cluster_dir)
                #Guardar el CSV con la información de imágenes y sus clusters
                csv_path = os.path.join(self.get_cluster_run_path(r), "cluster_images.csv")
                df = pd.DataFrame(cluster_data, columns=["cluster", "img"])
                df.sort_values(by="cluster").to_csv(csv_path, index=False)
            except (os.error) as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)









    # ###############################################################################3
    # VISUALIZATION METHODS
    # ###############################################################################3



    def create_plots(self, runs):
        
        for i, run in runs.iterrows():
            if "silhouette" in self.eval_method:
                self.show_best_silhouette(run)
                self.show_best_scatter(run)
                self.show_best_scatter(run, keep_original_embeddings = False)
                self.show_best_scatter_with_centers(run)
                self.show_best_clusters_counters_comparision(run)
            elif "davies" in self.eval_method:
                self.show_best_scatter(run)
                self.show_best_scatter(run, keep_original_embeddings = False)
                self.show_best_scatter_with_centers(run)
                self.show_best_clusters_counters_comparision(run)
            else:
                raise ValueError("Eval Method not support for plotting")





    def show_best_silhouette(self, run):
        """
        Displays the top `top_n` clusters with the highest silhouette average and the 
        `top_n` clusters with the lowest silhouette average, only if the total cluster 
        count exceeds `min_clusters`. If there are `min_clusters` or fewer clusters, 
        it displays all clusters without filtering.
        """
        # Extract information from the run
        best_run = run
        best_id = best_run['params.id']
        best_labels = best_run['artifacts.labels']
        clustering = best_run['params.clustering']
        dim_red = best_run['params.dim_red']
        dimensions = best_run['params.dimensions']
        optimizer = best_run['params.optimizer']
        original_score = best_run['metrics.score_wo_penalty']
        embeddings_used = best_run['artifacts.embeddings']


        min_clusters = self.n_cluster_range[0]
        top_n = int(self.n_cluster_range[0]/2)


        # Exclude noise points (label -1)
        non_noise_mask = best_labels != -1
        non_noise_labels = best_labels[non_noise_mask]
        non_noise_data = embeddings_used[non_noise_mask]

        # Calculate silhouette values for non-noise data
        silhouette_values = silhouette_samples(non_noise_data, non_noise_labels)

        # Calculate average silhouette per cluster
        unique_labels = np.unique(non_noise_labels)
        cluster_count = len(unique_labels)

        # Determine top and bottom clusters
        top_clusters = []
        bottom_clusters = []
        if cluster_count <= min_clusters:
            selected_clusters = unique_labels
        else:
            cluster_silhouette_means = {
                label: silhouette_values[non_noise_labels == label].mean() for label in unique_labels
            }
            top_clusters = sorted(cluster_silhouette_means, key=cluster_silhouette_means.get, reverse=True)[:top_n]
            bottom_clusters = sorted(cluster_silhouette_means, key=cluster_silhouette_means.get)[:top_n]
            selected_clusters = sorted(set(top_clusters + bottom_clusters), key=lambda label: cluster_silhouette_means[label])

        # Setup the figure with GridSpec
        fig = plt.figure(figsize=(10, 12))
        spec = gridspec.GridSpec(2, 1, height_ratios=[7, 3])  # 70% for plot, 30% for legend

        # Generate a unique color palette for the selected clusters
        colors = sns.color_palette("tab20", len(selected_clusters))
        cluster_color_map = {label: colors[i] for i, label in enumerate(selected_clusters)}

        # Create the plot area (top 70%)
        ax_plot = fig.add_subplot(spec[0])
        y_lower = 10
        yticks = []  # To store Y-axis positions for cluster labels
        for i, label in enumerate(selected_clusters):
            ith_cluster_silhouette_values = silhouette_values[non_noise_labels == label]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # Fill the silhouette for each cluster with a unique color
            ax_plot.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=cluster_color_map[label],
                alpha=0.7,
                label=f"Cluster {label}"
            )
            yticks.append(y_lower + 0.5 * size_cluster_i)  # Position for the cluster label on the Y-axis
            y_lower = y_upper + 10

            if i == len(top_clusters) - 1:
                ax_plot.axhline(y=y_lower, color='black', linestyle='--', linewidth=1.5)

        # Add a vertical line for the original silhouette score
        ax_plot.axvline(x=original_score, color="red", linestyle="--", label=f"Original Score: {original_score:.3f}")
        ax_plot.set_xlabel("Silhouette Coefficient", fontsize=16)
        ax_plot.set_ylabel("Cluster Index", fontsize=16)
        ax_plot.set_title(f"Silhouette Plot for Exp. {best_id} - {optimizer}\n"
                        f"Clustering: {clustering} | Dim Reduction: {dim_red} | Dimensions: {dimensions}\n"
                        f"Silhouette: {original_score:.3f}", fontsize=18)
        ax_plot.set_yticks(yticks)  # Set Y-axis ticks
        ax_plot.set_yticklabels(selected_clusters)  # Label the Y-axis ticks with cluster indices

        # Create the legend area (bottom 30%)
        ax_legend = fig.add_subplot(spec[1])
        ax_legend.axis("off")  # Hide the axes for the legend area
        handles, labels = ax_plot.get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5))
        ax_legend.legend(
            handles[:len(selected_clusters)],  # Only include handles for the selected clusters
            labels[:len(selected_clusters)],  # Corresponding labels
            loc="center",
            title="Most Representative Clusters Top/Bottom Clusters",
            fontsize='small',
            title_fontsize='small',
            ncol=4
        )

        # Save and optionally show the plot
        file_suffix = "best_silhouette"
        file_path = os.path.join(self.get_cluster_run_path(best_run),f"{file_suffix}.png")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, bbox_inches="tight")

        logger.info(f"Silhouette plot saved to {file_path}.")







    def show_best_scatter(self, run, keep_original_embeddings=True):
        """
        Plots a 2D scatter plot for the best run configuration, with clusters reduced 
        to 2D space using PCA and color-coded for better visual distinction. Points labeled 
        as noise (-1) are always shown in red.
        """

        best_run = run
        best_id = best_run['params.id']
        best_labels = np.array(best_run['artifacts.labels'])
        clustering = best_run['params.clustering']
        dim_red = best_run['params.dim_red']
        dimensions = best_run['params.dimensions']
        optimizer = best_run['params.optimizer']
        embeddings_used = best_run['artifacts.embeddings']
        eval_method = best_run['params.eval_method']
        score = best_run['metrics.score_wo_penalty'] if eval_method in ("silhouette","davies_bouldin") else best_run['metrics.score_w_penalty']
        cluster_count = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)  # Exclude noise (-1) from cluster count

        # Get original embeddings (avoind reduction over reduction embeddings)
        if keep_original_embeddings:
            data_df = pd.DataFrame(self.original_embeddings)
            data = data_df.values
        else:
            data = embeddings_used.values

        # Check if reduction is needed
        if data.shape[1] > 2:
            # If shape > 1, we cannot use selected reduction params, cause it doesnt make sense
            if dim_red == "umap":
                reducer = umap.UMAP(random_state=42, n_components=2, min_dist=0.2, n_neighbors=15)
                reduced_data = reducer.fit_transform(data)
            elif dim_red == "tsne":
                reducer = TSNE(random_state=42, n_components=2)
                reduced_data = reducer.fit_transform(data)
            else:
                pca = PCA(n_components=2, random_state=42)
                reduced_data = pca.fit_transform(data)
        else:
            # Use the data directly if already 2D
            reduced_data = data

        # Define colormap for clusters and manually assign red for noise
        colors = sns.color_palette("viridis", cluster_count)
        cmap = ListedColormap(colors)
        
        plt.figure(figsize=(12, 9))
        
        # Plot noise points (label -1) in red
        noise_points = reduced_data[best_labels == -1]
        plt.scatter(noise_points[:, 0], noise_points[:, 1], c='red', s=10, alpha=0.6, label="Noise (-1)")
        
        # Plot cluster points
        cluster_points = reduced_data[best_labels != -1]
        cluster_labels = best_labels[best_labels != -1]
        scatter = plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=cluster_labels, cmap=cmap, s=10, alpha=0.6)

        # Add colorbar if useful to distinguish clusters
        plt.colorbar(scatter, spacing="proportional", ticks=np.linspace(0, cluster_count, num=10))
        
        plt.title(f"Scatter Plot for Exp. {best_id} - {optimizer} (Noise in Red, Clusters in 2D) \n\n"
                  f"Clustering: {clustering} | Dim Reduction: {dim_red} | Dimensions: {dimensions}\n"
                  f"{eval_method}: {score:.3f}", fontsize=18)


        plt.xlabel("Component 1", fontsize=16)
        plt.ylabel("Component 2", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right',fontsize=14)

        # Save and show plot
        file_suffix = "best_scatter_original_embeddings" if keep_original_embeddings else "best_scatter_reduced_embeddings"
        file_path = os.path.join(self.get_cluster_run_path(best_run),f"{file_suffix}.png")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path, bbox_inches='tight')
        logger.info(f"Scatter plot generated for the selected run saved to {file_path}.")




    def show_best_scatter_with_centers(self, run):
        """
        Plots a 2D scatter plot for the best run configuration, with clusters reduced 
        to 2D space using PCA and color-coded for better visual distinction. Points labeled 
        as noise (-1) are always shown in red.
        """
        best_run = run

        best_labels = np.array(best_run['artifacts.labels'])
        best_id = best_run['params.id']
        eval_method = best_run['params.eval_method']
        best_centers = best_run['artifacts.centers'].values if isinstance(best_run['artifacts.centers'], pd.DataFrame) else np.array(best_run['artifacts.centers'])
        best_labels = best_run['artifacts.labels']
        clustering = best_run['params.clustering']
        dim_red = best_run['params.dim_red']
        dimensions = best_run['params.dimensions']
        optimizer = best_run['params.optimizer']
        score = best_run['metrics.score_wo_penalty'] if eval_method in ("silhouette","davies_bouldin") else best_run['metrics.score_w_penalty']
        embeddings_used = best_run['artifacts.embeddings']
        cluster_count = len(np.unique(best_labels)) - (1 if -1 in best_labels else 0)  # Exclude noise (-1) from cluster count

        # Get data reduced from eda object
        data = embeddings_used.values

        # Check if reduction is needed
        if data.shape[1] > 2:
            # If shape > 1, we cannot use selected reduction params, cause it doesnt make sense
            if dim_red == "umap":
                reducer = umap.UMAP(random_state=42, n_components=2, min_dist=0.2, n_neighbors=15)
                reduced_data = reducer.fit_transform(data)
                pca_centers = reducer.transform(best_centers)
            elif dim_red == "tsne":
                reducer = TSNE(random_state=42, n_components=2)
                reduced_data = reducer.fit_transform(data)
                pca_centers = reducer.transform(best_centers)
            else:
                reducer = PCA(n_components=2, random_state=42)
                reduced_data = reducer.fit_transform(data)
                pca_centers = reducer.transform(best_centers)
        else:
            # Use the data directly if already 2D
            reduced_data = data
            pca_centers = best_centers


        # Color mapping for clusters and plot setup
        colors = ['#00FF00', '#FFFF00', '#0000FF', '#FF9D0A', '#00B6FF', '#F200FF', '#FF6100']
        cmap_bold = ListedColormap(colors)
        plt.figure(figsize=(12,9))
        
        # Plot noise points (label -1) in red
        noise_points = reduced_data[best_labels == -1]

        print(type(best_labels), type(reduced_data))
        print(best_labels.shape, reduced_data.shape)

        plt.scatter(noise_points[:, 0], noise_points[:, 1], c='red', s=10, alpha=0.6, label="Noise (-1)")
        
        # Plot cluster points
        cluster_points = reduced_data[best_labels != -1]
        cluster_labels = best_labels[best_labels != -1]
        scatter = plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=cluster_labels, cmap=cmap_bold, s=15, alpha=0.6)

        # Plot cluster centers
        if pca_centers is not None:
            plt.scatter(pca_centers[:, 0], pca_centers[:, 1], marker='D', c='black', s=10, label="Cluster Centers", edgecolors='black')
        
        # Add colorbar to distinguish clusters
        plt.colorbar(scatter, spacing="proportional", ticks=np.arange(0, cluster_count + 1, max(1, cluster_count // 10)))

        
        plt.title(f"Scatter Plot for Exp. {best_id} - {optimizer} (Noise in Red) \n\n"
                f"Clustering: {clustering} | Dim Reduction: {dim_red} | Dimensions: {dimensions}\n"
                f"{eval_method}: {score:.3f}", fontsize=18)
        plt.xlabel("Component 1", fontsize=16)
        plt.ylabel("Component 2", fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right',fontsize=14)


        # Save and show plot
        file_suffix = "best_scatter_with_centers_reduced"
        file_path = os.path.join(self.get_cluster_run_path(best_run),f"{file_suffix}.png")
        plt.savefig(file_path, bbox_inches='tight')

        logger.info(f"Scatter plot generated for the selected run saved to {file_path}.")




    def show_best_clusters_counters_comparision(self,  run):
        """
        Displays a bar chart comparing the number of points in each cluster for the best configuration.
        
        The method retrieves the cluster sizes (number of points per cluster) from `label_counter`
        for the best run configuration and displays a bar chart to compare cluster sizes.

        Parameters
        ----------
        show_plots : bool, optional
            If True, displays the plot. Default is False.
        """
        best_run = run

        best_labels = np.array(best_run['artifacts.labels'])
        best_id = best_run['params.id']
        eval_method = best_run['params.eval_method']
        best_labels = best_run['artifacts.labels']
        eval_method = best_run['params.eval_method']
        score = best_run['metrics.score_wo_penalty'] if eval_method in ("silhouette","davies_bouldin") else best_run['metrics.score_w_penalty']
        label_counter = best_run['artifacts.label_counter']
        
        label_counter_filtered = {k: v for k, v in label_counter.items() if k != -1}

        # Extract cluster indices and their respective counts from label_counter
        cluster_indices = list(label_counter_filtered.keys())
        cluster_sizes = list(label_counter_filtered.values())

        # Count total with noise and without noise
        total_minus_one = label_counter.get(-1, 0)
        total_rest = sum(v for k, v in label_counter.items() if k != -1)
        
        # Plot the bar chart
        plt.figure(figsize=(12, 6))
        sns.barplot(x=cluster_indices, y=cluster_sizes, palette="viridis")
        plt.xlabel("Cluster Index")
        plt.ylabel("Number of Points")
        plt.title(f"Comparison of Cluster Sizes for Exp. {best_id}\n\n" \
                  f"Total cluster points: {total_rest}\n"   \
                  f"Total noise points: {total_minus_one}\n" \
                  f"{eval_method}: {score:.3f}", fontsize=18)
        step = 10
        cluster_indices.sort()
        plt.xticks(ticks=range(0, len(cluster_indices), step), labels=[cluster_indices[i] for i in range(0, len(cluster_indices), step)], rotation=90)
        
        # Save the plot with a name based on the `experiment` type
        file_suffix = "clusters_counter_comparison"
        file_path = os.path.join(self.get_cluster_run_path(best_run),f"{file_suffix}.png")
        plt.savefig(file_path, bbox_inches='tight')

        logger.info(f"Scatter plot generated for the selected run saved to {file_path}.")








if __name__ == "__main__":

    reduction_params = {
        "n_components": (2,25),
        "n_neighbors": (3,60),
        "min_dist": (0.1, 0.8)
    }
    n_cluster_range = (40,500)
    
    experiment_controller = ExperimentResultController("silhouette", 
                                                           13,
                                                           "small",
                                                           experiment_name="1_test_small_umap_silhouette", 
                                                           n_cluster_range=n_cluster_range,
                                                           reduction_params=reduction_params)
    best_run = experiment_controller.get_best_run()
    print(best_run)

