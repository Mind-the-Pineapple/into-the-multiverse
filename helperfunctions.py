from itertools import product

import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
import bct
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import hypergeom, spearmanr
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from tqdm import tqdm

# Set the random seed
np.random.seed(2)
rng = np.random.default_rng(2)


def gateway_coef_sign(W, ci, centrality_type='degree'):
    '''
    The gateway coefficient is a variant of participation coefficient.
    It is weighted by how critical the connections are to intermodular
    connectivity (e.g. if a uode is the only connection between its
    module and another module, it will have a higher gateway coefficient,
    unlike participation coefficient).
    Parameters
    ----------
    W : NxN np.ndarray
        undirected signed connection matrix
    ci : Nx1 np.ndarray
        community affiliation vector
    centrality_type : enum
        'degree' - uses the weighted degree (i.e, node strength)
        'betweenness' - uses the betweenness centrality
    Returns
    -------
    Gpos : Nx1 np.ndarray
        gateway coefficient for positive weights
    Gneg : Nx1 np.ndarray
        gateway coefficient for negative weights
    Reference:
        Vargas ER, Wahl LM, Eur Phys J B (2014) 87:1-10

    Note
    ----
    This function was copied from the bctpy package. The main diffrence is that
    line 84 was commented out to avoid unnecessary printing.
    '''
    _, ci = np.unique(ci, return_inverse=True)
    ci += 1
    n = len(W)
    np.fill_diagonal(W, 0)

    def gcoef(W):
        #strength
        s = np.sum(W, axis=1)
        #neighbor community affiliation
        Gc = np.inner((W != 0), np.diag(ci))
        #community specific neighbors
        Sc2 = np.zeros((n,))
        #extra modular weighting
        ksm = np.zeros((n,))
        #intra modular wieghting
        centm = np.zeros((n,))

        if centrality_type == 'degree':
            cent = s.copy()
        elif centrality_type == 'betweenness':
            cent = bct.betweenness_wei(invert(W))

        nr_modules = int(np.max(ci))
        for i in range(1, nr_modules+1):
            ks = np.sum(W * (Gc == i), axis=1)
            #print(np.sum(ks))
            Sc2 += ks ** 2
            for j in range(1, nr_modules+1):
                #calculate extramodular weights
                ksm[ci == j] += ks[ci == j] / np.sum(ks[ci == j])

            #calculate intramodular weights
            centm[ci == i] = np.sum(cent[ci == i])

        #print(Gc)
        #print(centm)
        #print(ksm)
        #print(ks)

        centm = centm / max(centm)
        #calculate total weights
        gs = (1 - ksm * centm) ** 2

        Gw = 1 - Sc2 * gs / s ** 2
        Gw[np.where(np.isnan(Gw))] = 0
        Gw[np.where(np.logical_not(Gw))] = 0

        return Gw

    G_pos = gcoef(W * (W > 0))
    G_neg = gcoef(-W * (W < 0))
    return G_pos, G_neg


def get_dissimilarity_n_neighbours(all_neighbours_orig,
                                   all_neighbours_reduced):
    '''
    Calculate the dissimilarity

    Parameters
    ----------
    all_neighbours_orig:
    all_neighbours_reduced:

    Returns
    -------
    all_dissimilarity: Dissimilarity scores between the original and reduced
    space
    '''

    all_dissimilarity = []
    for K in range(len(all_neighbours_reduced)):
        # Find the set of different indices
        diff = set(sorted(all_neighbours_orig[K])) - \
               set(sorted(all_neighbours_reduced[K]))
        # Calculate the dissimilarity
        epsilon = len(diff) / len(all_neighbours_orig[K])
        all_dissimilarity.append(epsilon)

    return all_dissimilarity


def get_models_neighbours(N, n_neighbors_step, data):
    '''
    Calculate the dissimilarity

    Parameters
    ----------
    n_neighbours: number of neighbours to analyse
    data: data (pairwise_subjects, n_analysis)

    Returns
    -------
    all_dissimilarity: Dissimilarity scores between the original and reduced
    space
    '''
    n_neighbours = range(2, N, n_neighbors_step)
    all_adj = np.zeros((len(data), len(data), len(n_neighbours)))
    all_neighbours_orig = []

    for idx, n_neighbour in enumerate(n_neighbours):
        adj = kneighbors_graph(data, n_neighbour, mode='distance',
                            metric='euclidean')
        adj_array = adj.toarray()
        all_adj[:, :, idx] = adj_array
        nneighbours_orig = np.nonzero(adj_array)
        nneighbours_orig = [item for item in zip(nneighbours_orig[0],
                                                 nneighbours_orig[1])]
        all_neighbours_orig.append(nneighbours_orig)

    return all_neighbours_orig, all_adj


def get_null_distribution(N, n_neighbors_step):
    # Calculate the null distribution using binary distribution
    def expectation(N, K):
        rv = hypergeom(N, K, K)
        x = np.arange(0, K)
        pmf = rv.pmf(x)
        return np.sum(x*pmf)

    null_distribution = []
    for K in range(2, N, n_neighbors_step):
        E = expectation(N, K)
        diss = 1 - (E/K)
        null_distribution.append(diss)
    return null_distribution


def objectiveFunc(TempModelNum, Y, Sparsities_Run, Data_Run, BCT_models, BCT_Run,
                  CommunityIDs, data1, data2, svr_params):
    '''

    Define the objective function for the Bayesian optimization.  This consists
    of the number indicating which model to test, a count variable to help
    control which subjects are tested, a random permutation of the indices of the
    subjects, the predictor variables and the actual y outcomes, the number of
    subjects to include in each iteration

    Parameters
    ----------
        TempModelNum: idx of the analysis being run
        Y: Y variable that will be predicted
        Sparsities_Run: List of threshold used
        Data_Run: Data used for creating the space
        BCT_models: Dictionary containing the list of models used
        BCT_Run: List containing the order in which the BCT models were run
        CommunityIDs: Information about the Yeo network Ids
        data1: Motion Regression functional connectivity data of the subjects
               that were not used to create the space
        data2: Global Signal Regression data for the subjects that gridsearch.cv_results['mean_test_score']were not used
               to create the space
        ClassOrRegress: Define if it is a classification or regression problem
        (0: classification; 1 regression)

    Returns
    -------
        score: Returns the MAE of the predictions
    '''

    TotalRegions = 346
    if Data_Run[TempModelNum] == 'MRS':
        TempData = data1
    elif Data_Run[TempModelNum] == 'GRS':
        TempData = data2
    else:
        ValueError('This type of pre-processing is not supported')
    TotalSubjects = TempData.shape[2]

    TempThreshold = Sparsities_Run[TempModelNum]
    BCT_Num = BCT_Run[TempModelNum]

    TempResults = np.zeros([TotalSubjects, TotalRegions])
    for SubNum in range(0, TotalSubjects):
        x = bct.threshold_proportional(TempData[:, :, SubNum],
                                       TempThreshold, copy=True)
        if BCT_Num == 'local efficiency':
            ss = BCT_models[BCT_Num](x, 1);
        elif BCT_Num == 'modularity (louvain)':
            temp = BCT_models[BCT_Num](x);
            ss = temp[0]
        elif BCT_Num== 'modularity (probtune)':
            temp = BCT_models[BCT_Num](x);
            ss = temp[0]
        elif BCT_Num == 'participation coefficient':
            ss = BCT_models[BCT_Num](x, CommunityIDs);
        elif BCT_Num == 'module degree z-score':
            ss = BCT_models[BCT_Num](x, CommunityIDs);
        elif BCT_Num == 'pagerank centrality':
            ss = BCT_models[BCT_Num](x, 0.85)
        elif BCT_Num == 'diversity coefficient':
            temp = BCT_models[BCT_Num](x, CommunityIDs)
            ss = temp[0]
        elif BCT_Num == 'gateway degree':
            temp = BCT_models[BCT_Num](x, CommunityIDs)
            ss = temp[0]
        elif BCT_Num == 'k-core centrality':
            temp = BCT_models[BCT_Num](x)
            ss = temp[0]
        else:
            ss = BCT_models[BCT_Num](x)
        #For each subject for each approach keep the 346 regional values.
        TempResults[SubNum, :] = ss

    RandInt = np.random.randint(10000)
    #svr_c = svr_params[TempModelNum]['svr__C']
    model = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
    cv = KFold(n_splits=10, shuffle=True, random_state=RandInt)
    scores = cross_val_score(model, TempResults, Y.ravel(), cv=cv,
                             scoring='neg_mean_absolute_error')

    # Note: the scores were divided by 10 in order to keep the values close
    # to 0 for avoiding problems with the Bayesian Optimisation
    # Load the pre-trained model
    score = scores.mean()/10
    return score


def posterior(gp, x_obs, y_obs, z_obs, grid_X):
    xy = (np.array([x_obs.ravel(), y_obs.ravel()])).T
    gp.fit(xy, z_obs)
    mu, std = gp.predict(grid_X.reshape(-1, 2), return_std=True)
    return mu, std, gp


# Helper function for calculating posterior predictions only for points
# in the space where an analysis approach exists
def posteriorOnlyModels(gp, x_obs, y_obs, z_obs, AllModelEmb):
    xy = (np.array([x_obs.ravel(), y_obs.ravel()])).T
    gp.fit(xy, z_obs)
    mu, std = gp.predict(AllModelEmb, return_std=True)
    return mu, std, gp


def display_gp_mean_uncertainty(kernel, optimizer, pbounds, BadIter):
    '''
    Code to display the estimated GP regression mean across the space as well
    as the uncertainty, showing which points were sampled.
    This is based on Pedro's code
    '''
    x = np.linspace(pbounds['b1'][0] - 10, pbounds['b1'][1] + 10, 50).reshape(
        -1, 1)
    y = np.linspace(pbounds['b2'][0] - 10, pbounds['b2'][1] + 10, 50).reshape(
        -1, 1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=10)
    #x_obs = np.array([[res["params"]["b1"]] for res in optimizer.res])
    #y_obs = np.array([[res["params"]["b2"]] for res in optimizer.res])
    #z_obs = np.array([res["target"] for res in optimizer.res])
    x_temp = np.array([[res["params"]["b1"]] for res in optimizer.res])
    y_temp = np.array([[res["params"]["b2"]] for res in optimizer.res])
    z_temp = np.array([res["target"] for res in optimizer.res])

    x_obs=x_temp[BadIter==0]
    y_obs=y_temp[BadIter==0]
    z_obs=z_temp[BadIter==0]

    x1x2 = np.array(list(product(x, y)))
    X0p, X1p = x1x2[:, 0].reshape(50, 50), x1x2[:, 1].reshape(50, 50)

    mu, sigma, gp = _posterior(gp, x_obs, y_obs, z_obs, x1x2)

    Zmu = np.reshape(mu, (50, 50))
    Zsigma = np.reshape(sigma, (50, 50))

    conf0 = np.array(mu - 2 * sigma).reshape(50, 50)
    conf1 = np.array(mu + 2 * sigma).reshape(50, 50)

    fig = plt.figure(figsize=(23, 23))
    X0p, X1p = np.meshgrid(x, y,indexing='ij')

    font_dict_title = {'fontsize': 25}
    font_dict_label = {'fontsize': 18}
    font_dict_label3 = {'fontsize': 15}

    ax0 = fig.add_subplot(321)
    fig0 = ax0.pcolormesh(X0p, X1p, Zmu)
    ax0.set_title('Gaussian Process Predicted Mean', fontdict=font_dict_title)
    ax0.set_xlabel('Component 1', fontdict=font_dict_label)
    ax0.set_ylabel('Component 2', fontdict=font_dict_label)
    fig.colorbar(fig0)

    ax1 = fig.add_subplot(322)
    fig1 = ax1.pcolormesh(X0p, X1p, Zsigma)
    ax1.set_title('Gaussian Process Variance', fontdict=font_dict_title)
    ax1.set_xlabel('Component 1', fontdict=font_dict_label)
    ax1.set_ylabel('Component 2', fontdict=font_dict_label)
    fig.colorbar(fig1)

    ax2 = fig.add_subplot(323, projection='3d')
    fig2 = ax2.plot_surface(X0p, X1p, Zmu, label='prediction',
                            cmap=cm.coolwarm)
    ax2.set_title('Gaussian Process Mean', fontdict=font_dict_title)
    ax2.set_xlabel('Component 1', fontdict=font_dict_label3)
    ax2.set_ylabel('Component 2', fontdict=font_dict_label3)
    ax2.set_zlabel('P. Mean', fontdict=font_dict_label3)

    ax3 = fig.add_subplot(324, projection='3d')
    fig3 = ax3.plot_surface(X0p, X1p, Zsigma, cmap=cm.coolwarm)
    ax3.set_title('Gaussian Process Variance', fontdict=font_dict_title)
    ax3.set_xlabel('Component 1', fontdict=font_dict_label3)
    ax3.set_ylabel('Component 2', fontdict=font_dict_label3)
    ax3.set_zlabel('Variance', fontdict=font_dict_label3)

    ax4 = fig.add_subplot(325, projection='3d')
    fig4 = ax4.plot_surface(X0p, X1p, conf0, label='confidence', alpha=0.3)
    fig4 = ax4.plot_surface(X0p, X1p, conf1, label='confidence', alpha=0.3)
    ax4.set_title('95% Confidence Interval', fontdict=font_dict_title)
    ax4.set_xlabel('Component 1', fontdict=font_dict_label3)
    ax4.set_ylabel('Component 2', fontdict=font_dict_label3)
    ax4.set_zlabel('P.Mean', fontdict=font_dict_label3)

    plt.show()
    fig.savefig('BOptResults1.png')

    return gp

def initialize_bo(ModelEmbedding, kappa, repetitions=False, DiffInit=None):
    """
    """
    if repetitions:
        RandomSeed = 118 + DiffInit
    else:
        RandomSeed = 118
    np.random.seed(RandomSeed)

    # Define the kernel: white noise kernel plus Mattern
    kernel = 1.0 * Matern(length_scale=25, length_scale_bounds=(10,80),nu=2.5) \
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 0.1))

    # Define bounds
    lb1 = np.min(ModelEmbedding[:, 0])
    hb1 = np.max(ModelEmbedding[:, 0])
    lb2 = np.min(ModelEmbedding[:, 1])
    hb2 = np.max(ModelEmbedding[:, 1])
    pbounds = {'b1': (lb1, hb1), 'b2': (lb2, hb2)}

    # For finding nearest point in space to next suggested sample from
    # Bayesian optimization
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree'
            ).fit(ModelEmbedding)

    # Acquisition function. Larger k (exploratory) smaller k (exploitatory)
    utility = UtilityFunction(kind="ucb", kappa=kappa, xi=1e-1)

    # Number of burn in random initial samples
    init_points = 10
    # Number of iterations of Bayesian optimization after burn in
    if repetitions:
        n_iter = 10
    else:
        n_iter = 40

    # Initialise optimizer
    optimizer = BayesianOptimization(f=None,
                                     pbounds=pbounds,
                                     verbose=4,
                                     random_state=RandomSeed)

    optimizer.set_gp_params(kernel=kernel, normalize_y=True,
                            n_restarts_optimizer=10)
    return kernel, optimizer, utility, init_points, n_iter, pbounds, nbrs, \
           RandomSeed

def run_bo(kernel, optimizer, utility, init_points, n_iter,
           pbounds, nbrs, RandomSeed, ModelEmbedding, BCT_models,
           BCT_Run, Sparsities_Run, Data_Run, Ages, CommunityIDs,
           data1, data2, ClassOrRegress, MultivariateUnivariate=True,
           repetitions=False, verbose=True):

    BadIters = np.empty(0)
    LastModel = -1
    Iter = 0
    if repetitions:
        pbar = tqdm(total=(init_points) + n_iter)
    else:
        pbar = tqdm(total=(2 * init_points) + n_iter)
    while Iter < init_points + n_iter:
        np.random.seed(RandomSeed+Iter)
        # If burnin
        if Iter < init_points:
            # Choose point in space to probe next in search space randomly
            next_point_to_probe = {'b1': np.random.uniform(pbounds['b1'][0],
                                                           pbounds['b1'][1]),
                                   'b2': np.random.uniform(pbounds['b2'][0],
                                                           pbounds['b2'][1])}
            if verbose:
                print("Next point to probe is:", next_point_to_probe)
            s1, s2 = next_point_to_probe.values()

        # if optimization
        else:
            # Choose point in space to probe next in search space using optimizer
            next_point_to_probe = optimizer.suggest(utility)
            if verbose:
                print("Next point to probe is:", next_point_to_probe)
            s1, s2 = next_point_to_probe.values()
        # convert suggested coordinates to np array
        Model_coord = np.array([[s1, s2]])
        # find the index of the models that are closest to this point
        distances, indices = nbrs.kneighbors(Model_coord)

        # I order to reduce repeatedly sampling the same point, check if
        # suggested point was sampled last and then check in ModelNums what the
        # name/index of that model is, if was recently sampled then take the
        # second nearest point.
        if LastModel == np.asscalar(indices[0][0]):
            TempModelNum = np.asscalar(indices[0][1])
            ActualLocation = ModelEmbedding[np.asscalar(indices[0][1])]
            Distance=distances[0][1]
        else:
            TempModelNum = np.asscalar(indices[0][0])
            ActualLocation = ModelEmbedding[np.asscalar(indices[0][0])]
            Distance=distances[0][0]

        if (Distance <10 or Iter<init_points):
            #Hack: because space is continuous but analysis approaches aren't,
            # we penalize points that are far (>10 distance in model space)
            #from any actual analysis approaches by assigning them the value of
            # the worst performing approach in the burn-in
            LastModel = TempModelNum
            BadIters = np.append(BadIters,0)
            # Call the objective function and evaluate the model/pipeline
            if MultivariateUnivariate:
                target = objectiveFunc(TempModelNum, Ages, Sparsities_Run,
                                       Data_Run, BCT_models, BCT_Run,
                                       CommunityIDs, data1, data2,
                                       ClassOrRegress)
                if verbose:
                    print("Next Iteration")
                    print(Iter)
                    # print("Model Num %d " % TempModelNum)
                    print('Print indices: %d  %d' % (indices[0][0], indices[0][1]))
                    print(Distance)
                    print("Target Function: %.4f" % (target))
                    print(' ')
                np.random.seed(Iter)
                # This is a hack. Add a very small random number to the coordinates so
                # that even if the model has been previously selected the GP thinks its
                # a different point, since this was causing it to crash
                TempLoc1 = ActualLocation[0] + (np.random.random_sample(1) - 0.5)/10
                TempLoc2 = ActualLocation[1] + (np.random.random_sample(1) - 0.5)/10
                pbar.update(1)
        else:
            newlist = sorted(optimizer.res, key=lambda k: k['target'])
            target=newlist[0]['target']
            LastModel = -1

            if verbose:
                print("Next Iteration")
                print(Iter)
                # print("Model Num %d " % TempModelNum)
    #            print('Print indices: %d  %d' % (indices[0][0], indices[0][1]))
                print(Distance)
                print("Target Function Default Bad: %.4f" % (target))
                print(' ')

            BadIters = np.append(BadIters,1)
            TempLoc1 = Model_coord[0][0]
            TempLoc2 = Model_coord[0][1]
            n_iter = n_iter+1

        Iter = Iter+1

        # Update the GP data with the new coordinates and model performance
        register_sample = {'b1': TempLoc1, 'b2': TempLoc2}
        optimizer.register(params=register_sample, target=target)
    pbar.close()
    return BadIters


def plot_bo_estimated_space(kappa, BadIter, optimizer, pbounds, ModelEmbedding,
                        PredictedAcc, kernel, output_path):
    x = np.linspace(pbounds['b1'][0] - 10, pbounds['b1'][1] + 10, 500).reshape(
    -1, 1)
    y = np.linspace(pbounds['b2'][0] - 10, pbounds['b2'][1] + 10, 500).reshape(
        -1, 1)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=10)

    x_temp = np.array([[res["params"]["b1"]] for res in optimizer.res])
    y_temp = np.array([[res["params"]["b2"]] for res in optimizer.res])
    z_temp = np.array([res["target"] for res in optimizer.res])

    x_obs=x_temp[BadIter==0]
    y_obs=y_temp[BadIter==0]
    z_obs=z_temp[BadIter==0]

    NumSamplesToInclude=x_obs.shape[0]
    x1x2 = np.array(list(product(x, y)))
    X0p, X1p = x1x2[:, 0].reshape(500, 500), x1x2[:, 1].reshape(500, 500)

    mu, sigma, gp = posterior(gp, x_obs[0:NumSamplesToInclude],
                              y_obs[0:NumSamplesToInclude],
                              z_obs[0:NumSamplesToInclude], x1x2)

    Zmu = np.reshape(mu, (500, 500))
    Zsigma = np.reshape(sigma, (500, 500))

    conf0 = np.array(mu - 2 * sigma).reshape(500, 500)
    conf1 = np.array(mu + 2 * sigma).reshape(500, 500)

    X0p, X1p = np.meshgrid(x, y, indexing='ij')

    font_dict_title = {'fontsize': 25}
    font_dict_label = {'fontsize': 15}
    font_dict_label3 = {'fontsize': 15}
    #print(x_obs.shape)
    vmax = Zmu.max()
    vmin = Zmu.min()

    cm = ['coolwarm', 'seismic']
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,8))

    ax = ax1
    pcm = ax.pcolormesh(X0p, X1p, Zmu, vmax=vmax, vmin=vmin, cmap=cm[0],
                        rasterized=True)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal', 'box')
    ax = ax2
    pcm = ax.scatter(ModelEmbedding[0:PredictedAcc.shape[0],0],
                     ModelEmbedding[0:PredictedAcc.shape[0],1],
                     c=PredictedAcc*10, vmax=vmax*10, vmin=vmin*10,
                     cmap=cm[0], rasterized=True)
    ax.set_aspect('equal', 'box')

    fig.tight_layout()
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.825, 0.35, 0.02, 0.3])
    fig.colorbar(pcm, cax=cbar_ax)

    fig.savefig(str(output_path / f'BOptAndTrueK{kappa}.png'), dpi=300)
    fig.savefig(str(output_path / f'BOptAndTrueK{kappa}.svg'), format='svg', dpi=300)

    return x_obs, y_obs, z_obs, x, y, gp, vmax, vmin

# TODO: Check if ModelDict has the same pipeline as fitted_model
# Save the results in a pickle

def plot_bo_evolution(kappa, x_obs, y_obs, z_obs, x, y, gp, vmax, vmin,
                      ModelEmbedding, PredictedAcc, output_path):
    fig, axs = plt.subplots(5, 3, figsize=(12,18))
    n_samples = [5,10,20,30,50]
    cm = ['coolwarm', 'seismic']
    PredictedAcc = PredictedAcc * 10
    for idx, NumSamplesToInclude in enumerate(n_samples):

        x1x2 = np.array(list(product(x, y)))
        X0p, X1p = x1x2[:, 0].reshape(500, 500), x1x2[:, 1].reshape(500, 500)
        mu, sigma, gp = posterior(gp, x_obs[0:NumSamplesToInclude],
                                   y_obs[0:NumSamplesToInclude],
                                   z_obs[0:NumSamplesToInclude], x1x2)
        muModEmb,sigmaModEmb,gpModEmb = posteriorOnlyModels(gp,
                                                          x_obs[0:NumSamplesToInclude],
                                                          y_obs[0:NumSamplesToInclude],
                                                          z_obs[0:NumSamplesToInclude],
                                                          ModelEmbedding)
        Zmu = np.reshape(mu, (500, 500))
        Zsigma = np.reshape(sigma, (500, 500))

        conf0 = np.array(mu - 2 * sigma).reshape(500, 500)
        conf1 = np.array(mu + 2 * sigma).reshape(500, 500)

        X0p, X1p = np.meshgrid(x, y, indexing='ij')

        ax = axs[idx,0]

        pcm = ax.pcolormesh(X0p, X1p, Zmu,vmax=vmax, vmin=vmin,
                cmap=cm[0],rasterized=True)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax = axs[idx,1]

        pcm = ax.pcolormesh(X0p, X1p, Zsigma,cmap=cm[1],rasterized=True)#,vmax=vmax,vmin=vmin)
        ax.set_title("Iterations: %i" % (NumSamplesToInclude),fontsize=15,fontweight="bold")
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)

        ax = axs[idx,2]
        # For visualisation purposes
        muModEmb = muModEmb * 10
        pcm=ax.scatter(muModEmb[PredictedAcc!=PredictedAcc.min()],
                       PredictedAcc[PredictedAcc!=PredictedAcc.min()],
                       marker='.', c='gray')
        ax.set_xlim(PredictedAcc.max(), PredictedAcc.min())
        ax.set_ylim(PredictedAcc.max(), PredictedAcc.min())
        #ax.set_xlim(-2.55, -2.25)
        #ax.set_ylim(-2.55, -2.25)
        ax.set_aspect('equal', 'box')

    fig.savefig(str(output_path / f'BOptEvolutionK{kappa}.svg'),format='svg',dpi=300)

    corr = spearmanr(muModEmb,PredictedAcc)
    return corr

def analysis_space(BCT_Num, BCT_models, x, KeptYeoIDs):
    if BCT_Num == 'local efficiency':
        ss = BCT_models[BCT_Num](x,1)
    elif BCT_Num == 'modularity (louvain)':
        ss, _ = BCT_models[BCT_Num](x, seed=2)
    elif BCT_Num== 'modularity (probtune)':
        ss, _ = BCT_models[BCT_Num](x, seed=2)
    elif BCT_Num == 'participation coefficient':
        ss = BCT_models[BCT_Num](x, KeptYeoIDs)
    elif BCT_Num == 'module degree z-score':
        ss = BCT_models[BCT_Num](x, KeptYeoIDs)
    elif BCT_Num == 'pagerank centrality':
        ss = BCT_models[BCT_Num](x, 0.85)
    elif BCT_Num == 'diversity coefficient':
        ss, _ = BCT_models[BCT_Num](x, KeptYeoIDs)
    elif BCT_Num == 'gateway degree':
        ss, _ = BCT_models[BCT_Num](x, KeptYeoIDs)
    elif BCT_Num == 'k-core centrality':
        ss, _ = BCT_models[BCT_Num](x)
    else:
        ss = BCT_models[BCT_Num](x)
    return ss

def plot_bo_repetions(ModelEmbedding, PredictedAcc, BestModelGPSpaceModIndex,
                      BestModelEmpiricalModIndex, BestModelEmpirical,
                      ModelActualAccuracyCorrelation, output_path):
    # displaying results of 20 iterations

    fig8 = plt.figure(constrained_layout=False,figsize=(18,6))
    gs1 = fig8.add_gridspec(nrows=6, ncols=18)
    ax1 = fig8.add_subplot(gs1[:,0:6])
    ax1.set_title('Optima GP regression: 20 iterations',fontsize=15,
            fontweight="bold")
    ax1.scatter(ModelEmbedding[0:PredictedAcc.shape[0],0],
                ModelEmbedding[0:PredictedAcc.shape[0],1],
                c=PredictedAcc*10,cmap='coolwarm',alpha=0.2,s=120)#vmax=vmax,vmin=vmin,
    ax1.scatter(ModelEmbedding[BestModelGPSpaceModIndex.astype(int)][:,0],
                ModelEmbedding[BestModelGPSpaceModIndex.astype(int)][:,1],s=120,c='black')

    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-50, 50)

    ax2 = fig8.add_subplot(gs1[:,7:13])
    ax2.set_title('Empirical optima: 20 iterations',fontsize=15,fontweight="bold")
    ax2.scatter(ModelEmbedding[0:PredictedAcc.shape[0],0],
                ModelEmbedding[0:PredictedAcc.shape[0],1],
                c=PredictedAcc*10,cmap='coolwarm',s=120,alpha=0.2)#vmax=vmax,vmin=vmin,
    ax2.scatter(ModelEmbedding[BestModelEmpiricalModIndex.astype(int)][:,0],
                ModelEmbedding[BestModelEmpiricalModIndex.astype(int)][:,1],
                c='black', s=120)

    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)

    ax3 = fig8.add_subplot(gs1[:, 14:16])
    ax3.violinplot([PredictedAcc*10, BestModelEmpirical*10])
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels(['Accuracy \n of all points', 'Accuracy\n of optima'],
            fontsize=9)

    ax4 = fig8.add_subplot(gs1[:,17:18])
    ax4.violinplot([ModelActualAccuracyCorrelation])
    ax4.set_xticks([1])
    ax4.set_xticklabels(['Correlation: \n est vs emp '],fontsize=9)

    fig8.savefig(str(output_path / 'BOpt20Repeats.png'),dpi=300)
    fig8.savefig(str(output_path / 'BOpt20Repeats.svg'),format="svg")
