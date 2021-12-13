#!/usr/bin/env python
# coding: utf-8

# # 1. Setting up the enviroment

# In[ ]:


# Install necessary python dependencies
get_ipython().system(' pip install -r requirements.txt')


# In[26]:


from pathlib import Path
from collections import OrderedDict
import pickle
import json
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor

from nilearn.connectome import ConnectivityMeasure
from umap.umap_ import UMAP
import phate

from helperfunctions import (initialize_bo, run_bo, load_abide_demographics, plot_bo_estimated_space, plot_bo_evolution,
                             posteriorOnlyModels, plot_bo_repetions, objective_func_class)

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Define key variables
# Add the into-the-multiverse folder to the Python path. This allows the helperfunction
# to be used
import sys
sys.path.insert(1, 'into-the-multiverse')
import numpy as np
np.random.seed(1234)
from pathlib import Path
import os


# In[ ]:


# Define paths - REMOTE
PROJECT_ROOT = Path.cwd()
data_path = PROJECT_ROOT / 'into-the-multiverse' /'data' / 'abide'
output_path = PROJECT_ROOT / 'into-the-multiverse' / 'output'/ 'abide'

# Change accordingly to where the raw data has been downloaded. The data can be downloaded from
# (http://preprocessed-connectomes-project.org/abide/download.html is downloaded
data_raw = Path('/Volumes/abide')
if not output_path.is_dir():
    output_path.mkdir(parents=True)


# In[2]:


# Define the space variables
derivatives = ['rois_tt', 'rois_ho', 'rois_ez', 'rois_dosenbach160', 'rois_cc400', 'rois_cc200']
pipelines = ['cpac', 'ccs', 'dparsf', 'niak']
strategies = ['filt_global', 'nofilt_global', 'nofilt_noglobal', 'filt_noglobal']
conn_metrics = ['tangent', 'correlation', 'partial correlation', 'covariance']


# # 2. Run the different analysis to bild the space

# The next step assumes that the data has been downloaded. The data can be downloaded from (http://preprocessed-connectomes-project.org/abide/download.html). For time reasons, we will not download the data within this notebook. To run this script the code expects the files to be in the following structure:
# 
# ```
# ├── ccs
# │   ├── filt_global
# │   ├── filt_noglobal
# │   ├── nofilt_global
# │   └── nofilt_noglobal
# ├── cpac
# │   ├── filt_global
# │   ├── filt_noglobal
# │   ├── nofilt_global
# │   └── nofilt_noglobal
# ├── dparsf
# │   ├── filt_global
# │   ├── filt_noglobal
# │   ├── nofilt_global
# │   └── nofilt_noglobal
# └── niak
#     ├── filt_global
#     ├── filt_noglobal
#     ├── nofilt_global
#     └── nofilt_noglobal
# ```
# 
# However, to facilitate reproducibility together with this code. We are providing the file `output/abide/abide_space.pckl`, which contains the output from the next cell. 

# In[3]:


# select the subjects we want to use to create the space (about 20% of the total subjects) making sure that
# both classes are equally represented

# Load data demographics
abide_df = load_abide_demographics(data_root)
indices = np.arange(len(abide_df))
idx_space, idx_train_holdout = train_test_split(indices, test_size=.8, train_size=.2, random_state=0,
                                        shuffle=True, stratify=abide_df['DX_GROUP'])
# Split the training data again, to keep part of the dataset as a hold out dataset

idx_train, idx_holdout = train_test_split(idx_train_holdout, test_size=.25, train_size=.75, random_state=0,
                                          shuffle=True, stratify=abide_df['DX_GROUP'].iloc[idx_train_holdout])
# Visualise stratification
space_df = abide_df.iloc[idx_space]
print('Numbers on space df')
print(space_df['DX_GROUP'].value_counts())

train_df = abide_df.iloc[idx_train]
print('Numbers on training df')
print(train_df['DX_GROUP'].value_counts())

holdout_df = abide_df.iloc[idx_holdout]
print('Numbers on hold out df')
print(holdout_df['DX_GROUP'].value_counts())

# save list of indexes of the data split
indices = {'idx_train': idx_train.tolist(),
           'idx_space': idx_space.tolist(),
           'idx_holdout': idx_holdout.tolist()}
with open((output_path / f'indices_space_train.json'), 'w') as handle:
    json.dump(indices, handle)


# The next cell will create the space

# In[ ]:


n_idx_space = int(len(idx_space) * (len(idx_space) - 1) / 2)
count = 0
ResultsIndVar = np.zeros(((len(derivatives) * len(pipelines) * len(strategies) * len(conn_metrics)), n_idx_space))
methods_idx = {}
space_rois = {}
with tqdm(range(len(derivatives) * len(pipelines) * len(strategies) * len(conn_metrics))) as pbar:
    for derivative in derivatives:
        space_rois[derivative] = {}
        for pipeline in pipelines:
            space_rois[derivative][pipeline] = {}
            for strategy in strategies:
                space_rois[derivative][pipeline][strategy] = {}
                for conn_metric in conn_metrics:
                    data_path = data_root / 'Outputs' / pipeline / strategy / derivative
                    space_rois[derivative][pipeline][strategy][conn_metric] = []
                    for subject_idx in idx_space:
                        subject = abide_df.iloc[subject_idx]['FILE_ID']
                        subject_path = data_path / f'{subject}_{derivative}.1D'
                        rois = pd.read_csv(subject_path, delimiter='\t')
                        space_rois[derivative][pipeline][strategy][conn_metric].append(rois.to_numpy())
                        methods_idx[count] = [derivative, pipeline, strategy, conn_metric]
                    count += 1
                    pbar.update(1)

count = 0
# Iterate over the possible configurations and calculate the connectivity metric.
with tqdm(range(len(derivatives) * len(pipelines) * len(strategies) * len(conn_metrics))) as pbar:
    for derivative in derivatives:
        for pipeline in pipelines:
            for strategy in strategies:
                for conn_metric in conn_metrics:
                    space_flat_rois = []
                    correlation_measure = ConnectivityMeasure(kind=conn_metric)
                    correlation_matrix = correlation_measure.fit_transform(
                        space_rois[derivative][pipeline][strategy][conn_metric])
                    # Plot the upper diagonal connectivity matrix, excluding the diagonal (k=1)
                    # correlation_matrix = np.triu(correlation_matrix, k=1)
                    # plotting.plot_matrix(correlation_matrix, colorbar=True, vmax=1, vmin=-1)
                    # plt.savefig(output_path / f'{subject}_{derivative}.png')
                    for subject_idx in range(len(idx_space)):
                        tmp = correlation_matrix[subject_idx][np.triu_indices(
                            space_rois[derivative][pipeline][strategy][conn_metric][0].shape[1], k=1)]
                        space_flat_rois.append(tmp)

                    # Build an array of similarities between subjects for each analysis approach. This is used as a
                    # distance metric between the different subjects
                    cos_sim = cosine_similarity(space_flat_rois)
                    ResultsIndVar[count, :] = cos_sim[np.triu_indices(len(idx_space), k=1)]
                    count += 1
                    pbar.update(1)

# Save results
save_results = {'Results': ResultsIndVar, 'methods_idx': methods_idx}
with open((output_path / 'abide_space.pckl'), 'wb') as handle:
    pickle.dump(save_results, handle)


# # 3. Building and analysing the low-dimensional space

# In[5]:


# Load the indices we want to use for the analysis
with open((output_path / f'indices_space_train.json'), 'r') as handle:
    indices = json.load(handle)

idx_train = indices['idx_train']
idx_space = indices['idx_space']

train_df = abide_df.iloc[idx_train]
print('Numbers on training df')
print(train_df['DX_GROUP'].value_counts())
space_df = abide_df.iloc[idx_space]
print('Numbers on space df')
print(space_df['DX_GROUP'].value_counts())


# In[6]:


with open((output_path / 'abide_space.pckl'), 'rb') as handle:
    save_results = pickle.load(handle)
ResultsIndVar = save_results['Results']
methods_idx = save_results['methods_idx']

# Reduced dataset
data_reduced = {}

# plot tSNE
Results = ResultsIndVar
scaler = StandardScaler()
X = scaler.fit_transform(Results.T)
X = X.T
n_neighbors = 60
n_components = 2
#Define different dimensionality reduction techniques
methods = OrderedDict()
LLE = partial(manifold.LocallyLinearEmbedding,
              n_neighbors, n_components, eigen_solver='dense')
methods['LLE'] = LLE(method='standard', random_state=0)
methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                           n_neighbors=n_neighbors, random_state=0)
methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',  perplexity=150,
                                 random_state=0)
methods['UMAP'] = UMAP(random_state=40, n_components=2, n_neighbors=200,
                             min_dist=.8)
methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=10,
                              random_state=21, metric=True)
methods['PHATE'] = phate.PHATE()


methods['PCA'] = PCA(n_components=2)


# In[7]:


# Define markers for the derivatives
markers = ['s', 'o', '^', 'D', 'v', '*']
markers_order = np.array([pip[0] for pip in methods_idx.values()])

# Define colors and markers for the pipeliens
#colourmaps = {'ccs': 'Greens', 'cpac': 'Purples', 'dparsf': 'Blues', 'niak': 'Reds'}
colourmaps = {'correlation': 'Greens', 'covariance': 'Purples', 'partial correlation': 'Blues', 'tangent': 'Reds'}
metrics_order = np.array([pip[3] for pip in methods_idx.values()])

# Define colors and markers for the strategies
markers_strategies = {'filt_global': .7, 'nofilt_global': .4, 'nofilt_noglobal': .15, 'filt_noglobal': .55}
strategies_order = [pip[2] for pip in methods_idx.values()]
strategies_int = np.array([markers_strategies[x] for x in strategies_order])

markers_metric = ['-', '/', '.', "x"]
markers_map = {'cpac': '-', 'ccs': '/', 'dparsf': '.', 'niak': 'x'}
pipeline_order = np.array([pip[1] for pip in methods_idx.values()])


# In[8]:


selected_analysis = 'MDS'
Lines = {}
Y = methods[selected_analysis].fit_transform(X)
data_reduced[selected_analysis] = Y
figMDS = plt.figure(figsize=(21, 15))
gsMDS = figMDS.add_gridspec(nrows=15, ncols=20)
axs = figMDS.add_subplot(gsMDS[:, 0:15])
#for idx_pip, pipeline in enumerate(sorted(colourmaps)):
for idx_metric, conn_metric in enumerate(sorted(colourmaps)):
    for idx_pipeline, pipeline in enumerate(sorted(pipelines)):
        for idx_derivative, derivative in enumerate(sorted(derivatives)):
            axs.scatter(Y[:, 0][(markers_order == derivative) & (metrics_order == conn_metric) & (pipeline_order == pipeline)],
                        Y[:, 1][(markers_order == derivative) & (metrics_order == conn_metric) & (pipeline_order == pipeline)],
                        c=strategies_int[(markers_order == derivative) & (metrics_order == conn_metric) & (pipeline_order == pipeline)],
                        s=180, marker=markers[idx_derivative], hatch=4*markers_metric[idx_pipeline],
                        norm=plt.Normalize(vmin=0, vmax=1),
                        cmap=colourmaps[conn_metric])
            Lines[idx_derivative] = mlines.Line2D([], [], color='black', linestyle='None', marker=markers[idx_derivative],
                                              markersize=10, label=derivative)
axs.spines['top'].set_linewidth(1.5)
axs.spines['right'].set_linewidth(1.5)
axs.spines['bottom'].set_linewidth(1.5)
axs.spines['left'].set_linewidth(1.5)
axs.set_xlabel('dimension 2', fontsize=25)
axs.set_ylabel('dimension 1', fontsize=25)
axs.tick_params(labelsize=15)
axs.set_title(f'{selected_analysis}', fontsize=20, fontweight="bold")
plt.axis('tight')
GreenPatch = mpatches.Patch(color='#52b365', label='correlation')
PurplePatch = mpatches.Patch(color='#8a86bf', label='covariance')
BluesPatch = mpatches.Patch(color='#4f9bcb', label='partial correlation')
RedsPatch = mpatches.Patch(color='#f34a36', label='tangent')
IntensityPatch1 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='filter and GSR',
                                 alpha=1)
IntensityPatch2 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='filter and no GSR',
                                 alpha=0.5)
IntensityPatch3 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='no filter and GSR',
                                 alpha=0.2)
IntensityPatch4 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='no filter and no GSR',
                                 alpha=0.1)
line_patchPatch = mpatches.Patch(facecolor=[0.1, 0.1, 0.1], hatch=4*markers_metric[0], label=sorted(pipelines)[0],
                                 alpha=.1)
dot_patchPatch = mpatches.Patch(facecolor=[0.1, 0.1, 0.1], hatch=4*markers_metric[1], label=sorted(pipelines)[1],
                                alpha=.1)
diagonal_patchPatch = mpatches.Patch(facecolor=[0.1, 0.1, 0.1], hatch=4*markers_metric[2], label=sorted(pipelines)[2],
                                     alpha=.1)
x_patchPatch = mpatches.Patch(facecolor=[0.1, 0.1, 0.1], hatch=4*markers_metric[3], label=sorted(pipelines)[3],
                                     alpha=.1)
BlankLine = mlines.Line2D([], [], linestyle='None')

plt.legend(handles=[GreenPatch, BluesPatch, PurplePatch, RedsPatch, BlankLine, IntensityPatch1,
                       IntensityPatch2, IntensityPatch3, IntensityPatch4, BlankLine,
                       Lines[0], Lines[1], Lines[2], Lines[3], Lines[4], Lines[5], BlankLine,
                       line_patchPatch, dot_patchPatch, diagonal_patchPatch, x_patchPatch
                     ],
           fontsize=24, frameon=False, bbox_to_anchor=(1.4, .97), bbox_transform=axs.transAxes)
plt.savefig(output_path / f'{selected_analysis}_v2.png',  dpi=300)
plt.savefig(output_path / f'{selected_analysis}_v2.svg', format='svg')


# In[9]:


# Plot the other methods
# Reduced dimensions
# As we already analysed the MDS drop it from the dictionary
methods.pop(selected_analysis)
gsDE, axs = plt.subplots(3, 2, figsize=(16, 16), constrained_layout=True)
axs = axs.ravel()
for idx_method, (label, method) in enumerate(methods.items()):
    Y = method.fit_transform(X)
    # Save the results
    data_reduced[label] = Y
    Lines = {}

    # for idx_pip, pipeline in enumerate(sorted(colourmaps)):
    for idx_metric, conn_metric in enumerate(sorted(colourmaps)):
        for idx_pipeline, pipeline in enumerate(sorted(pipelines)):
            for idx_derivative, derivative in enumerate(sorted(derivatives)):
                axs[idx_method].scatter(Y[:, 0][(markers_order == derivative) & (metrics_order == conn_metric) & (
                            pipeline_order == pipeline)],
                            Y[:, 1][(markers_order == derivative) & (metrics_order == conn_metric) & (
                                        pipeline_order == pipeline)],
                            c=strategies_int[(markers_order == derivative) & (metrics_order == conn_metric) & (
                                        pipeline_order == pipeline)],
                            s=180, marker=markers[idx_derivative], hatch=4 * markers_metric[idx_pipeline],
                            norm=plt.Normalize(vmin=0, vmax=1),
                            cmap=colourmaps[conn_metric])
                Lines[idx_derivative] = mlines.Line2D([], [], color='black', linestyle='None',
                                                      marker=markers[idx_derivative],
                                                      markersize=10, label=derivative)
    if idx_method %2 == 0:
        axs[idx_method].set_xlabel('Dimension 1', fontsize=20)
    if (idx_method == 4) or (idx_method == 5):
        axs[idx_method].set_ylabel('Dimension 2', fontsize=20)

    axs[idx_method].set_title(f'{label}', fontsize=20, fontweight="bold")
    axs[idx_method].axis('tight')
    axs[idx_method].tick_params(labelsize=15)

GreenPatch = mpatches.Patch(color='#52b365', label='correlation')
PurplePatch = mpatches.Patch(color='#8a86bf', label='covariance')
BluesPatch = mpatches.Patch(color='#4f9bcb', label='partial correlation')
RedsPatch = mpatches.Patch(color='#f34a36', label='tangent')
IntensityPatch1 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='filter and GSR',
                                 alpha=1)
IntensityPatch2 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='filter and no GSR',
                                 alpha=0.5)
IntensityPatch3 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='no filter and GSR',
                                 alpha=0.2)
IntensityPatch4 = mpatches.Patch(color=[0.1, 0.1, 0.1], label='no filter and no GSR',
                                 alpha=0.1)
line_patchPatch = mpatches.Patch(facecolor=[0.1, 0.1, 0.1], hatch=4 * markers_metric[0], label=sorted(pipelines)[0],
                                 alpha=.1)
dot_patchPatch = mpatches.Patch(facecolor=[0.1, 0.1, 0.1], hatch=4 * markers_metric[1], label=sorted(pipelines)[1],
                                alpha=.1)
diagonal_patchPatch = mpatches.Patch(facecolor=[0.1, 0.1, 0.1], hatch=4 * markers_metric[2],
                                     label=sorted(pipelines)[2],
                                     alpha=.1)
x_patchPatch = mpatches.Patch(facecolor=[0.1, 0.1, 0.1], hatch=4 * markers_metric[3], label=sorted(pipelines)[3],
                              alpha=.1)
BlankLine = mlines.Line2D([], [], linestyle='None')

gsDE.legend(handles=[GreenPatch, BluesPatch, PurplePatch, RedsPatch, BlankLine, IntensityPatch1,
                    IntensityPatch2, IntensityPatch3, IntensityPatch4, BlankLine,
                    Lines[0], Lines[1], Lines[2], Lines[3], Lines[4], Lines[5], BlankLine,
                    line_patchPatch, dot_patchPatch, diagonal_patchPatch, x_patchPatch],
           fontsize=15, frameon=False, bbox_to_anchor=(1.25, 0.7))

gsDE.savefig(str(output_path / 'dim_reduction.png'), dpi=300)
gsDE.savefig(str(output_path / 'dim_reduction.svg'), format='svg')


# In[10]:


gsDE.savefig(str(output_path / 'dim_reduction.png'), dpi=300, bbox_inches='tight')
gsDE.savefig(str(output_path / 'dim_reduction.svg'), format='svg', bbox_inches='tight')


# In[11]:


# save embeddings
with open((output_path / 'embeddings.pckl'), 'wb') as handle:
    pickle.dump(data_reduced, handle)


# # 4. Exhaustive Search

# As in step 1. this step also assumes that the data has been previously downloaded. If for computational purposes you do not want to download the data and re-calculate the predictions, we provide the exhaustively search spaced: `output/abide/predictedAcc.pckl`
# 
# Note: This is also a time consuming step and might take about 28hrs to complete

# In[ ]:


# Load the embedding results
with open((output_path / 'embeddings.pckl'), 'rb') as handle:
    embeddings = pickle.load(handle)
# Load the labels for the analysis
with open(output_path / 'abide_space.pckl', 'rb') as handle:
    Data_Run = pickle.load(handle)
# Load indices of the subjects used for train and test
with open((output_path / f'indices_space_train.json'), 'rb') as handle:
    indices = json.load(handle)

# TODO: make this more generalisable. We will use the MDS space
model_embedding = embeddings['MDS']

abide_df = load_abide_demographics(data_root)
# Select only models to train on
train_df = abide_df.iloc[indices['idx_train']]
train_labels = train_df['DX_GROUP']
files_id = train_df['FILE_ID']

PredictedAcc = np.zeros((len(Data_Run['Results'])))
for count in tqdm(range(len(Data_Run['Results']))):
    PredictedAcc[count] = objective_func_class(Data_Run['methods_idx'], count, train_labels, files_id,
                                                   data_raw, output_path)

# Dump predictions
pickle.dump(PredictedAcc, open(str(output_path / 'predictedAcc.pckl'), 'wb'))


# In[ ]:


plt.figure()
plt.scatter(model_embedding[0: PredictedAcc.shape[0], 0],
            model_embedding[0: PredictedAcc.shape[0], 1],
            c=(PredictedAcc), cmap='bwr')
plt.colorbar()
plt.savefig(output_path / 'Predictions.png')


# # 5. Active Learning

# Note: This step also requires the user to previously downalod the raw data. Due to computation limitations with colab, we are only providing the active learning without repetitions.

# In[ ]:


def compute_active_learning(kappa, model_config, CassOrRegression):
    # Load data demographics
    abide_df = load_abide_demographics(data_root)

    # Load the embedding results
    with open((output_path / 'embeddings.pckl'), 'rb') as handle:
        embeddings = pickle.load(handle)
    with open(output_path / 'abide_space.pckl', 'rb') as handle:
        Data_Run = pickle.load(handle)
    with open((output_path / 'predictedAcc.pckl'), 'rb') as handle:
        PredictedAcc = pickle.load(handle)
    model_embedding = embeddings['MDS']
    # Load indices of the subjects used for train and test
    with open((output_path / f'indices_space_train.json'), 'rb') as handle:
        indices = json.load(handle)
    # Remove subjects that were used to create the space
    train_df = abide_df.iloc[indices['idx_train']]
    Y = train_df['DX_GROUP']
    files_id = train_df['FILE_ID']

    # Check range of predictions
    PredictedAcc = pickle.load(open(str(output_path / "predictedAcc.pckl"), "rb"))
    print(f'Max {np.max(PredictedAcc)}')
    print(f'Min {np.min(PredictedAcc)}')
    print(f'Mean and std {np.mean(PredictedAcc)} and {np.std(PredictedAcc)}')

    model_config['Data_Run'] = Data_Run['methods_idx']
    model_config['files_id'] = train_df['FILE_ID']
    model_config['output_path'] = output_path

    kernel, optimizer, utility, init_points, n_iter, pbounds, nbrs, RandomSeed =         initialize_bo(model_embedding, kappa)

    BadIter = run_bo(optimizer, utility, init_points, n_iter,
           pbounds, nbrs, RandomSeed, model_embedding, model_config,
           Y, ClassOrRegression, MultivariateUnivariate=True,
           repetitions=False, verbose=True)

    x_exploratory, y_exploratory, z_exploratory, x, y, gp, vmax, vmin =         plot_bo_estimated_space(kappa, BadIter, optimizer, pbounds, model_embedding, PredictedAcc, kernel,
                                output_path, ClassOrRegression)

    corr = plot_bo_evolution(kappa, x_exploratory, y_exploratory, z_exploratory, x, y, gp,
                             vmax, vmin, model_embedding, PredictedAcc, output_path, ClassOrRegression)

    return corr
    


# In[ ]:


kappa = 10.0
# path to the raw data
model_config = {}
model_config['data_root'] = data_raw
ClassOrRegression = 'Classification'
corr = compute_active_learning(kappa, model_config, ClassOrRegression)
print(f'Spearman correlation {corr}')


# In[ ]:


kappa = .1
# path to the raw data
model_config = {}
model_config['data_root'] = data_raw
ClassOrRegression = 'Classification'
corr = compute_active_learning(kappa, model_config, ClassOrRegression)
print(f'Spearman correlation {corr}')


# ## Repetitions

# In[24]:


def calculate_conn(Y, files_id):
    TotalSubjects = len(Y)
    TempResults = []
    pipeline = Data_Run['methods_idx'][TempModelNum][1]
    strategy = Data_Run['methods_idx'][TempModelNum][2]
    derivative = Data_Run['methods_idx'][TempModelNum][0]
    data_path = data_root / 'Outputs' / pipeline / strategy / derivative
    # Load the data for every subject.
    for file_id in files_id:
        subject_path = data_path / f'{file_id}_{derivative}.1D'
        rois = pd.read_csv(subject_path, delimiter='\t')
        TempResults.append(rois.to_numpy())
    # Calculate the correlation using the selected metric
    correlation_measure = ConnectivityMeasure(kind=Data_Run['methods_idx'][TempModelNum][3])
    correlation_matrix = correlation_measure.fit_transform(TempResults)
    lower_diag_n = int(rois.shape[1] * (rois.shape[1] - 1) / 2)
    rois_l = np.zeros((TotalSubjects, lower_diag_n))
    for subject in range(TotalSubjects):
        rois_l[subject, :] = correlation_matrix[subject, :, :][np.triu_indices(rois.shape[1], k=1)]
    return rois_l


# In[21]:


# Load the embedding results
with open((output_path / 'embeddings.pckl'), 'rb') as handle:
    embeddings = pickle.load(handle)
# Load the labels for the analysis
with open(output_path / 'abide_space.pckl', 'rb') as handle:
    Data_Run = pickle.load(handle)
# Load indices of the subjects used for train and test
with open((output_path / f'indices_space_train.json'), 'rb') as handle:
    indices = json.load(handle)

# TODO: make this more generalisable. We will use the MDS space
model_embedding = embeddings['MDS']
kappa = 10

train_df = abide_df.iloc[indices['idx_train']]
train_Y = train_df['DX_GROUP']
train_files_id = train_df['FILE_ID']
holdout_df = abide_df.iloc[indices['idx_holdout']]
holdout_y = holdout_df['DX_GROUP']
holdout_files_id = holdout_df['FILE_ID']

ClassOrRegress = 'Classification'
model_config = {}
model_config['Data_Run'] = Data_Run['methods_idx']
model_config['files_id'] = train_df['FILE_ID']
model_config['data_root'] = data_root
model_config['output_path'] = output_path


# In[28]:


# Check range of predictions
PredictedAcc = pickle.load(open(str(output_path / "predictedAcc.pckl"), "rb"))
print(f'Max {np.max(PredictedAcc)}')
print(f'Min {np.min(PredictedAcc)}')
print(f'Mean and std {np.mean(PredictedAcc)} and {np.std(PredictedAcc)}')


# In[29]:


n_repetitions = 20

BestModelGPSpace = np.zeros(n_repetitions)
BestModelGPSpaceModIndex = np.zeros(n_repetitions)
BestModelEmpirical = np.zeros(n_repetitions)
BestModelEmpiricalModIndex = np.zeros(n_repetitions)
ModelActualAccuracyCorrelation = np.zeros(n_repetitions)
cv_scores = np.zeros(n_repetitions)

for DiffInit in range(n_repetitions):
    print(f'Repetiton #: {DiffInit}')
    # Define settings for the analysis
    kernel, optimizer, utility, init_points, n_iter, pbounds, nbrs, RandomSeed =         initialize_bo(model_embedding, kappa, repetitions=True,
                      DiffInit=DiffInit)

    FailedIters = run_bo(optimizer, utility, init_points,
                         n_iter, pbounds, nbrs, RandomSeed,
                         model_embedding, model_config, train_Y,
                         ClassOrRegress,
                         MultivariateUnivariate=True,
                         repetitions=True,
                         verbose=False)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=10)

    x_temp = np.array([[res["params"]["b1"]] for res in optimizer.res])
    y_temp = np.array([[res["params"]["b2"]] for res in optimizer.res])
    z_temp = np.array([res["target"] for res in optimizer.res])

    x_obs = x_temp[FailedIters == 0]
    y_obs = y_temp[FailedIters == 0]
    z_obs = z_temp[FailedIters == 0]

    muModEmb, sigmaModEmb, gpModEmb = posteriorOnlyModels(gp, x_obs, y_obs, z_obs,
                                                          model_embedding)

    BestModelGPSpace[DiffInit] = muModEmb.max()
    BestModelGPSpaceModIndex[DiffInit] = muModEmb.argmax()
    BestModelEmpirical[DiffInit] = z_obs.max()
    Model_coord = np.array([[x_obs[z_obs.argmax()][-1], y_obs[z_obs.argmax()][-1]]])
    BestModelEmpiricalModIndex[DiffInit] = nbrs.kneighbors(Model_coord)[1][0][0]
    ModelActualAccuracyCorrelation[DiffInit] = spearmanr(muModEmb, PredictedAcc)[0]
    TempModelNum = muModEmb.argmax()

    train_rois_l = calculate_conn(train_Y, train_files_id)
    holdout_rois_l = calculate_conn(holdout_y, holdout_files_id)

    model = Pipeline([('scaler', StandardScaler()), ('reg', LogisticRegression(penalty='l2', random_state=0))])

    model.fit(train_rois_l, train_Y.ravel())
    pred = model.predict(holdout_rois_l)
    y_proba = model.predict_proba(holdout_rois_l)[:, 1]
    score = roc_auc_score(holdout_y.ravel(), y_proba)

    #CVPValBestModels[DiffInit] = pvalue
    cv_scores[DiffInit] = score

df_best = pd.DataFrame(columns=['repetition', 'pipeline', 'derivatives', 'strategies', 'conn_metrics', 'score'])
for n in range(n_repetitions):
    n_results = {}
    n_results['repetition'] = n
    n_results['pipeline'] = Data_Run['methods_idx'][int(BestModelGPSpaceModIndex[n])][1]
    n_results['derivatives'] = Data_Run['methods_idx'][int(BestModelGPSpaceModIndex[n])][0]
    n_results['strategies'] = Data_Run['methods_idx'][int(BestModelGPSpaceModIndex[n])][2]
    n_results['conn_metrics'] = Data_Run['methods_idx'][int(BestModelGPSpaceModIndex[n])][3]
    n_results['score'] = cv_scores[n]
    df_best = df_best.append(n_results, ignore_index=True)
df_best = df_best.set_index('repetition')
# format the score column to a 3 digits
df_best['score'] = df_best['score'].apply('{:.3f}'.format)

repetions_results = {
    'dataframe': df_best,
    'BestModelGPSpaceModIndex': BestModelGPSpaceModIndex,
    'BestModelEmpiricalIndex': BestModelEmpiricalModIndex,
    'BestModelEmpirical': BestModelEmpirical,
    'ModelActualAccuracyCorrelation': ModelActualAccuracyCorrelation
}
pickle.dump(repetions_results, open(str(output_path / "repetitions_results.p"), "wb"))


# In[30]:


df_best


# In[ ]:




