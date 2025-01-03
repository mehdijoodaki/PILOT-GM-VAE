

from pilotpy.tools import *
import scanpy as sc
import time
from joblib import Parallel, delayed
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import random
import numpy as np
import torch
import time
from anndata import AnnData
import os
import tensorflow as tf
import scipy.linalg as spl
from joblib import Parallel, delayed
from numba import njit, prange
import scipy.linalg
import warnings
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.covariance import LedoitWolf
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from scipy.io import loadmat
from model.GMVAE import *
from model.GMVAE import GMVAE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from argparse import Namespace
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train_gmvae(
    adata,
    dataset_name,
    pca_key='X_pca',
    labels_column=None,
    train_proportion=0.8,
    batch_size=512,
    batch_size_val=200,
    seed=42,
    epochs=50,
    learning_rate=1e-3,
    decay_epoch=-1,
    lr_decay=0.5,
    gaussian_size=64,
    num_classes=11,
    input_size=None,
    init_temp=1.0,
    decay_temp=1,
    hard_gumbel=0,
    min_temp=0.5,
    decay_temp_rate=0.013862944,
    w_gauss=1.0,
    w_categ=1.0,
    w_rec=2.0,
    rec_type="mse",
    cuda=0,
    gpuID=0,
    verbose=0,
    save_model=True,
    load_weights=False,
    apply_gmm=True,
):
    """
    Train or load a GMVAE model, save it to a folder, and optionally perform inference.
    
    Parameters:
        adata (AnnData): The input data object.
        dataset_name (str): Name of the dataset for saving the model.
        pca_key (str): Key in `adata.obsm` for the PCA-transformed data.
        labels_column (str or None): Column name in `adata.obs` for labels.
        train_proportion (float): Proportion of data for training.
        batch_size (int): Training batch size.
        batch_size_val (int): Validation and test batch size.
        seed (int): Random seed.
        epochs (int): Total number of epochs for training.
        learning_rate (float): Learning rate for training.
        decay_epoch (int): Reduces learning rate every decay_epoch epochs.
        lr_decay (float): Learning rate decay factor.
        gaussian_size (int): Size of the Gaussian latent space.
        num_classes (int): Number of classes.
        input_size (int or None): Input size (e.g., PCA dimension); inferred if None.
        init_temp (float): Initial temperature for Gumbel-Softmax.
        decay_temp (int): Flag to decay Gumbel temperature each epoch.
        hard_gumbel (int): Flag for using hard Gumbel-Softmax.
        min_temp (float): Minimum Gumbel temperature after annealing.
        decay_temp_rate (float): Rate of temperature decay.
        w_gauss (float): Weight for Gaussian loss.
        w_categ (float): Weight for categorical loss.
        w_rec (float): Weight for reconstruction loss.
        rec_type (str): Type of reconstruction loss ('bce' or 'mse').
        cuda (bool): Whether to use GPU.
        gpuID (int): ID of the GPU to use.
        verbose (int): Verbosity level.
        save_model (bool): Save the model after training.
        load_weights (bool): Load pre-trained weights if available.
        apply_gmm (bool): applying gmm on weights.
    
    Returns:
       model

    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
      torch.cuda.manual_seed(seed)

    data = adata.obsm[pca_key]
    input_size = data.shape[1] if input_size is None else input_size
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    
    # Extract labels if provided
    if labels_column and labels_column in adata.obs:
        labels = adata.obs[labels_column].astype('category').cat.codes
        labels_tensor = torch.tensor(labels.values, dtype=torch.long)
        dataset = TensorDataset(data_tensor, labels_tensor)  # Dataset with data and labels
    else:
        labels_tensor = None
        # Create a dataset with only data (no labels)
        dataset = TensorDataset(data_tensor)  # Only data

    
    # Extract labels if provided
    if labels_column and labels_column in adata.obs:
        labels = adata.obs[labels_column].astype('category').cat.codes
        labels_tensor = torch.tensor(labels.values, dtype=torch.long)
        dataset = TensorDataset(data_tensor, labels_tensor)
    else:
        labels_tensor = torch.zeros(len(data_tensor), dtype=torch.long)
        dataset = TensorDataset(data_tensor, labels_tensor)
    
   
    # Function to partition dataset into train, validation, and test
    def partition_dataset(dataset, train_proportion=train_proportion):
        n = len(dataset)
        train_num = int(n * train_proportion)
        indices = np.random.permutation(n)
        train_indices, val_indices = indices[:train_num], indices[train_num:]
        return train_indices, val_indices
    
    
    train_indices, val_indices = partition_dataset(dataset, train_proportion)
    test_indices = val_indices[:len(val_indices) // 2]
    val_indices = val_indices[len(val_indices) // 2:]
    
    # Create DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=batch_size_val, sampler=SubsetRandomSampler(val_indices))
    test_loader = DataLoader(dataset, batch_size=batch_size_val, sampler=SubsetRandomSampler(test_indices))
    whole_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Prepare arguments for GMVAE
    args = Namespace(
        dataset=dataset_name,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        batch_size_val=batch_size_val,
        learning_rate=learning_rate,
        decay_epoch=decay_epoch,
        lr_decay=lr_decay,
        gaussian_size=gaussian_size,
        num_classes=num_classes,
        input_size=input_size,
        init_temp=init_temp,
        decay_temp=decay_temp,
        hard_gumbel=hard_gumbel,
        min_temp=min_temp,
        decay_temp_rate=decay_temp_rate,
        w_gauss=w_gauss,
        w_categ=w_categ,
        w_rec=w_rec,
        rec_type=rec_type,
        cuda=cuda,
        labels_column=labels_column,
        verbose=verbose,
    )
   
    # Initialize GMVAE
    gmvae = GMVAE(args)
    
    # Model directory and path
    model_dir = f"./trained_models/{dataset_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "gmvae_weights.pth")
    
    # Load pre-trained weights or train the model
    if load_weights:
        gmvae.network.load_state_dict(torch.load(model_path))
        gmvae.network.eval()  # Set the model to evaluation mode
        print(f"Loaded pre-trained weights from {model_path}.")
    else:
        # Train the model
        print("Training the GMVAE model...")
        gmvae.train(train_loader, val_loader)
        if save_model:
            torch.save(gmvae.network.state_dict(), model_path)
            print(f"Saved trained model weights to {model_path}.")
    
    # Perform inference
    print("Performing inference...")
    z_latent, x_recon, cluster_probs, clusters = gmvae.infer(whole_loader)
    
    adata.obsm['z_laten'] = z_latent
    adata.obsm['weights'] = cluster_probs
    adata.obsm['x_prim']=x_recon
    adata.obs['cluster_assignments_by_model_before_gmm']=clusters
    adata.obs['component_assignment'] = np.nan
    if apply_gmm:
        gmm_clusters = GaussianMixture(n_components=num_classes)
        cluster_assignments = gmm_clusters.fit_predict(cluster_probs)
        adata.obs['component_assignment'] = cluster_assignments
    else:
        adata.obs['component_assignment'] = clusters
        
    print("Done!")
    return gmvae




def plot_umap_and_stacked_bar(
    adata, 
    cell_type_col, 
    component_col='component_assignment', 
    palette_name=None, 
    title="Cell Type vs Component Assignment", 
    xlabel="Component Assignment", 
    ylabel="Cell Count", 
    save_path='figures/Bar_plot.png', 
    umap_save_name="_components_vs_cell_types.png", 
    umap_size=5, 
    umap_legend_fontsize=10, 
    umap_ncols=2, 
    umap_wspace=0.55, 
    bar_figsize=(14, 8), 
    umap_figsize=(10, 10), 
    axes_titlesize=18, 
    axes_labelsize=16, 
    legend_fontsize=14, 
    legend_bbox_to_anchor=(1.05, 1), 
    legend_loc='upper left', 
    legend_borderaxespad=0.0
):
    """
    Plots a UMAP and a stacked bar chart of component assignments vs. cell types.

    Parameters:
        adata: Input AnnData object.
        cell_type_col (str): Name of the column containing cell type annotations.
        component_col (str): Name of the column containing component assignments.
        palette_name (str, optional): Name of the palette or a custom palette list to use for coloring.
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        save_path (str, optional): File path to save the bar plot (PDF or PNG). If None, the plot is not saved.
        umap_save_name (str, optional): File name for saving the UMAP plot.
        umap_size (int, optional): Point size for the UMAP plot.
        umap_legend_fontsize (int, optional): Font size for the UMAP legend.
        umap_ncols (int, optional): Number of columns for the UMAP plots.
        umap_wspace (float, optional): Width space between UMAP plots.
        bar_figsize (tuple, optional): Figure size for the bar plot.
        umap_figsize (tuple, optional): Figure size for the UMAP plot.
        axes_titlesize (int, optional): Font size for axes titles.
        axes_labelsize (int, optional): Font size for axes labels.
        legend_fontsize (int, optional): Font size for the legend.
        legend_bbox_to_anchor (tuple, optional): Anchor position for the legend.
        legend_loc (str, optional): Location of the legend.
        legend_borderaxespad (float, optional): Padding between legend and axes border.

    Returns:
        None
    """
    # Ensure component_col is treated as a string
    adata.obs[component_col] = adata.obs[component_col].astype(str)

    # Set default palette if none is provided
    if palette_name is None:
        palette = sns.color_palette("Set1", 9) + sns.color_palette("Set2", 8) + sns.color_palette("Set3", 12) + sns.color_palette("Dark2", 4)
        palette = palette[:33]  # Limit the palette to exactly 33 colors
    else:
        palette = palette_name

    # Plot UMAP
    sc.pl.umap(
        adata,
        color=[component_col, cell_type_col],
        save=umap_save_name,  
        ncols=umap_ncols, 
        wspace=umap_wspace, 
        legend_fontsize=umap_legend_fontsize, 
        size=umap_size, 
        palette=palette
    )

    # Set font properties globally
    plt.rc('axes', titlesize=axes_titlesize)  # Set font size for axes titles
    plt.rc('axes', labelsize=axes_labelsize)  # Set font size for axes labels
    plt.rc('legend', fontsize=legend_fontsize)  # Set font size for legend

    # Prepare data for the stacked bar plot
    df = pd.DataFrame({
        'component_assignment': adata.obs[component_col],
        'Cell Type': adata.obs[cell_type_col]
    })

    # Create a cross-tab to get counts (no normalization)
    cross_tab = pd.crosstab(df[component_col], df['Cell Type'])

    # Ensure that the number of unique cell types matches the palette size
    num_cell_types = cross_tab.shape[1]
    if num_cell_types > len(palette):
        raise ValueError(f"Your palette has {len(palette)} colors, but you have {num_cell_types} unique cell types.")

    # Plot as a stacked bar plot using the custom palette (no normalization)
    plt.figure(figsize=bar_figsize)  # Use dynamic figure size for bar plot
    # Plot as a stacked bar plot using the custom palette (no normalization)
    ax = cross_tab.plot(kind='bar', stacked=True, color=palette[:num_cell_types], figsize=bar_figsize)

    # Add labels and titles
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Move legend to the right side of the plot
    plt.legend(
        title='Cell Type', 
        bbox_to_anchor=legend_bbox_to_anchor, 
        loc=legend_loc, 
        borderaxespad=legend_borderaxespad
    )

    # Save the plot as a file if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.tight_layout()
    plt.show()



def compute_distance(k, l, m_s, m_t, C_s, C_t, covariance_type, log=False,epsilon = 1e-3):
    """
    Computes the Bures-Wasserstein distance between components k and l of GMMs.
    
    Args:
    - k: index of component in GMM s.
    - l: index of component in GMM t.
    - m_s: mean vectors of GMM s (shape: [num_components_s, dim]).
    - m_t: mean vectors of GMM t (shape: [num_components_t, dim]).
    - C_s: covariance matrices of GMM s (shape: [num_components_s, dim] for diag, [num_components_s, dim, dim] for full).
    - C_t: covariance matrices of GMM t (shape: [num_components_t, dim] for diag, [num_components_t, dim, dim] for full).
    - covariance_type: either 'diag' for diagonal covariance or 'full' for full covariance.
    - log: whether to output log information in the Bures-Wasserstein computation.

    Returns:
    - Bures-Wasserstein distance between component k of GMM s and component l of GMM t.
    """
    
    ms_k = m_s[k, :]  # Mean vector of component k in GMM s
    mt_l = m_t[l, :]  # Mean vector of component l in GMM t

    if covariance_type == 'diag': 
        
        Cs_k = np.diag(C_s[k, :][k])   # Convert diagonal elements into a diagonal covariance matrix
        Ct_l = np.diag(C_t[l, :][l])  # Convert diagonal elements into a diagonal covariance matrix
         
    else:
        # Full covariance case
        Cs_k = C_s[k, :, :]  # Covariance matrix of component k in GMM s
        Ct_l = C_t[l, :, :]  # Covariance matrix of component l in GMM t
        Cs_k += epsilon * np.eye(Cs_k.shape[0])
        Ct_l += epsilon * np.eye(Ct_l.shape[0])

    attempt=0
    max_attempts=10
    bures_wasserstein = ot.gaussian.bures_wasserstein_distance(ms_k, mt_l, Cs_k, Ct_l, log=log)
    while attempt < max_attempts:
        if np.isnan(bures_wasserstein):
            epsilon *= 0.5
            Cs_k += epsilon * np.eye(Cs_k.shape[0])
            Ct_l += epsilon * np.eye(Ct_l.shape[0])
            bures_wasserstein = ot.gaussian.bures_wasserstein_distance(ms_k, mt_l, Cs_k, Ct_l, log=log)
            attempt=attempt+1
        else:
            break
      
    
    return bures_wasserstein


def compute_emd(i, j, samples_id, EMD, adata, compute_distance, ot,covariance_type,wass_dis,epsilon = 1e-3,log=False):
    s = samples_id[i]
    t = samples_id[j]
    if s == t:
        return None  # No need to compute for the same sample pair
    
    if EMD[i, j] != 0:
        return None  # Already computed this pair
    
    gmm_repr_s = adata.uns['GMVAE_Representation'][s]
    m_s = np.array(gmm_repr_s['means'])
    C_s = np.array(gmm_repr_s['covariances'])

    gmm_repr_t = adata.uns['GMVAE_Representation'][t]
    m_t = np.array(gmm_repr_t['means'])
    C_t = np.array(gmm_repr_t['covariances'])

    num_components_s = m_s.shape[0]
    num_components_t = m_t.shape[0]

    # Normalize weights
    weights1 = np.array(gmm_repr_s['weights']) / np.sum(gmm_repr_s['weights'])
    weights2 = np.array(gmm_repr_t['weights']) / np.sum(gmm_repr_t['weights'])

    # Compute distances in parallel for components
    distances_flat = Parallel(n_jobs=-1)(
        delayed(compute_distance)(k, l, m_s, m_t, C_s, C_t, covariance_type, log=log,epsilon =epsilon) 
        for k in range(num_components_s) for l in range(num_components_t)
    )
    distances = np.array(distances_flat).reshape(num_components_s, num_components_t)

    # Compute the EMD
    w_d = ot.emd2(weights1, weights2, distances, numThreads=50)

    # Store results symmetrically in the matrix
    EMD[i, j] = w_d
    EMD[j, i] = w_d

    return (i, j, w_d)  # Return the result to update the matrix later




def gmmvae_wasserstein_distance(adata,emb_matrix='X_PCA',
clusters_col='component_assignment',sample_col='sampleID',status='status',
                              metric='cosine',
                            
                               regulizer=0.2,normalization=True,
                               regularized='unreg',reg=0.1,
                               res=0.01,steper=0.01,data_type='scRNA',
                 return_sil_ari=False,random_state=2,covariance_type='full',wass_dis=True,epsilon = 1e-4,n_neighbors=15,resolution=0.5,log=False):

    num_components=len(adata.obs['component_assignment'].unique())
    data,annot=extract_data_anno_scRNA_from_h5ad(adata,emb_matrix=emb_matrix,
    clusters_col=clusters_col,sample_col=sample_col,status=status)
    pca_results_df = pd.DataFrame(adata.obsm[emb_matrix]).reset_index(drop=True)

    adata.uns['annot']=annot
    sample_ids = adata.obs[sample_col].reset_index(drop=True)
    cell_subtypes = adata.obs[clusters_col].reset_index(drop=True)
    status = adata.obs[status].reset_index(drop=True)
    
        # Concatenate the PCA results with 'sampleID', 'cell_subtype', and 'status'
    combined_pca_df = pd.concat([pca_results_df, sample_ids, cell_subtypes, status], axis=1)
  
    #combined_pca_df = combined_pca_df.rename(columns={clusters_col: 'cell_types',Status:'status'  }) 
                                            
    current_columns = combined_pca_df.columns

    # Create a mapping for the last three columns
    rename_dict = {current_columns[-3]: 'sampleID', 
                   current_columns[-2]: 'cell_type', 
                   current_columns[-1]: 'status'}
    
    # Rename the columns using the dictionary
    combined_pca_df.rename(columns=rename_dict, inplace=True)                                                                                            
        #combined_pca_df.columns[-3:]=['cell_type','sampleID','status']
    adata.uns['Datafame'] = combined_pca_df
    if wass_dis:
        gmmvae_Representation(adata, sample_col=sample_col,covariance_type=covariance_type,n_neighbors=n_neighbors,resolution=resolution,num_components=num_components)
        samples_id = list(adata.uns['GMVAE_Representation'].keys())
        n_samples = len(samples_id)
        EMD = np.zeros((n_samples, n_samples))
        
           # Extract means and covariances
        samples_id = list(adata.uns['GMVAE_Representation'].keys())
        n_samples = len(samples_id)
        EMD = np.zeros((n_samples, n_samples))
    #########################

    start_time = time.time()
    if wass_dis:
        
        start_time = time.time()
        # Parallelize the outer loops
        results = Parallel(n_jobs=-1)(
            delayed(compute_emd)(i, j, samples_id, EMD, adata, compute_distance, ot, covariance_type,wass_dis,epsilon=epsilon,log=log)
            for i in range(n_samples) for j in range(i + 1, n_samples)  # Only compute for j > i to avoid duplicates
        )

        # Update the EMD matrix with the computed distances
        for rest in results:
            if rest is not None:
                i, j, w_d = rest
                EMD[i, j] = w_d
                EMD[j, i] = w_d
   
    
        emd_df = pd.DataFrame.from_dict(EMD).T
        emd_df.columns=samples_id 
        emd_df['sampleID']=samples_id 
        emd_df=emd_df.set_index('sampleID')
        adata.uns['EMD_df']=emd_df
        adata.uns['EMD'] =EMD


    else:
        EMD=adata.uns['EMD']
   
    #Computing clusters and then ARI
    if return_sil_ari:
        predicted_labels, ARI, real_labels = Clustering(EMD, annot,metric=metric,res=res,steper=steper)
        adata.uns['real_labels'] =real_labels
        #Computing Sil
        Silhouette = Sil_computing(EMD, real_labels,metric=metric)
        adata.uns['Sil']=Silhouette
        adata.uns['ARI']=ARI
    else:
        adata.uns['real_labels']=return_real_labels(annot)
        
    
    annot= adata.uns['annot']
    #annot['component_assignment']=list(adata.obs['component_assignment'])
    proportions = Cluster_Representations(annot)
    adata.uns['proportions'] = proportions
   


                                  
      

def gmmvae_Representation(adata, num_components=5,sample_col='sampleID',
                                        covariance_type='full',
                                        n_neighbors=15,resolution=0.5):
    df = adata.uns['Datafame']
    df = df[df['cell_type'] != 'Unknown']
    df = df.drop(['status'], axis=1)
    df = df.drop(['cell_type'], axis=1)
    pca_data = df.drop(['sampleID'], axis=1).values  # Assume PCA features are here
    data_tensor = torch.tensor(pca_data, dtype=torch.float32)  # Convert to tensor

    input_dim = pca_data.shape[1]
    sources = df['sampleID'].unique()


    weights=adata.obsm['weights']
    
    cluster_assignments = adata.obs['component_assignment'] 
    # Calculate means and covariances for each sample
    params = {}
    for source in sources:
        data = df[df['sampleID'] == source]
        data_values = data.drop(['sampleID'], axis=1).values
        data_tensor_source = torch.tensor(data_values, dtype=torch.float32)

        # Get indices as a list for proper indexing
        source_indices = data.index.tolist()

        if source_indices:

            component_assignments = cluster_assignments[source_indices]
            adata_obs_indices = adata.obs.index[adata.obs[sample_col] == source].tolist()
            adata.obs.loc[adata_obs_indices, 'component_assignment'] = component_assignments
            # Initialize arrays to store means and covariances for each component
            means = np.zeros((num_components, input_dim))
            covariances = np.zeros((num_components, input_dim, input_dim))

            # Calculate means and covariances for each component
            for k in range(num_components):
                # Get the indices of the cells assigned to component k
                indices = component_assignments == k
                assigned_data = data_tensor_source[indices]

                if assigned_data.size(0) > 0:  # Ensure there are assigned points
                    means[k] = assigned_data.mean(dim=0).detach().numpy()
                    if covariance_type=='full':
                        covariances[k] = np.cov(assigned_data.numpy(), rowvar=False)
                    else:

                        covariances[k] = np.var(assigned_data.numpy(), axis=0)
                        covariances[k] = np.diag(covariances[k])

            # Mixing weights (proportions) for this sample
            weights_sample = weights[source_indices].mean(axis=0)

           # weights_sample = weights[source_indices].mean(dim=0).detach().numpy()

            params[source] = {
                'means': means,
                'covariances': covariances,
                'weights': weights_sample,
                'proportion': len(data_values) / len(df)
            }

    annot= adata.uns['annot']
    #annot['component_assignment']=list(adata.obs['component_assignment'])
    proportions = Cluster_Representations(annot)
    adata.uns['proportions'] = proportions
    adata.uns['GMVAE_Representation'] = params

    return adata

