from itertools import product

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import seed
from mpl_toolkits.mplot3d import Axes3D
import bct
from sklearn_rvm import EMRVR
from sklearn.svm import LinearSVR, LinearSVC, SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

def objectiveFunc(TempModelNum,Y,Sparsities_Run,Data_Run,BCT_models,BCT_Run,CommunityIDs,data1,data2,ClassOrRegress):
    '''

    Define the objective function for the Bayesian optimization.  This consists
    of the number indicating which model to test, a count variable to help
    control which subjects are tested, a random permutation of the indices of the
    subjects, the predictor variables and the actual y outcomes, the number of
    subjects to include in each iteration
    '''

    
    if Data_Run[TempModelNum]==0:
        TempData=data1
        TotalRegions=346
        TotalSubjects=TempData.shape[2]
    elif Data_Run[TempModelNum]==1:
        TempData=data2
        TotalRegions=346
        TotalSubjects=TempData.shape[2]
    

    
    TempThreshold=Sparsities_Run[TempModelNum]
    
    BCT_Num=[i for i, e in enumerate(BCT_models) if e[0] == BCT_Run[TempModelNum]][0]
    
    TempResults=np.zeros([TotalSubjects,TotalRegions])
    for SubNum in range(0,TotalSubjects):
        x = bct.threshold_proportional(TempData[:,:,SubNum], TempThreshold, copy=True)
        if BCT_Num==7:
            s=np.asarray(BCT_models[BCT_Num][1](x,1))
        elif BCT_Num==8:
            temp_s=np.asarray(BCT_models[BCT_Num][1](x))
            s=temp_s[0]
        elif BCT_Num==9:
            temp_s=np.asarray(BCT_models[BCT_Num][1](x))
            s=temp_s[0]
        elif BCT_Num==10:
            s=np.asarray(BCT_models[BCT_Num][1](x,CommunityIDs))
        elif BCT_Num==11:
            s=np.asarray(BCT_models[BCT_Num][1](x,CommunityIDs))
        elif BCT_Num==12:
            s=np.asarray(BCT_models[BCT_Num][1](x,0.85))
                #elif BCT_Num==13:
                #   temp_s=np.asarray(BCT_models[BCT_Num][1](x))
                #   s=temp_s[0]
        elif BCT_Num==13:
                #temp_s=np.asarray(BCT_models[BCT_Num][1](x,KeptYeoIDs,'degree'))
            temp_s=np.asarray(BCT_models[BCT_Num][1](x,CommunityIDs))
            s=temp_s[0] 
        elif BCT_Num==14:
            temp_s=np.asarray(BCT_models[BCT_Num][1](x,CommunityIDs))
            s=temp_s[0] 
        elif BCT_Num==15:
            temp_s=np.asarray(BCT_models[BCT_Num][1](x))
            s=temp_s[0]
        else:
            s=np.asarray(BCT_models[BCT_Num][1](x))

        TempResults[SubNum,:]=s 
    scaler = StandardScaler()
    TempResults=scaler.fit_transform(TempResults)

    


    RandInt=np.random.randint(10000)
    print(RandInt)
    #model = EMRVR(kernel="linear")
    if ClassOrRegress==1:
        #model = LinearSVR(random_state=0, tol=1e-5)
        model = SVR(C=1.0, epsilon=0.2)
        cv = KFold(n_splits=10, shuffle=True, random_state=RandInt)
        scores = cross_val_score(model, TempResults, Y.ravel(), cv=cv,scoring='neg_mean_absolute_error')
        score=scores.mean()/10
        
    else: 
        model = SVC(gamma='auto')
        cv = StratifiedKFold(n_splits=10, random_state=RandInt, shuffle=True)
        scores = cross_val_score(model, TempResults, Y.ravel(), cv=cv,scoring='neg_mean_absolute_error')
        score=scores.mean()/10
    
    
    return score


def _posterior(gp, x_obs, y_obs, z_obs, grid_X):
    xy = (np.array([x_obs.ravel(), y_obs.ravel()])).T
    gp.fit(xy, z_obs)
    mu, std = gp.predict(grid_X.reshape(-1, 2), return_std=True)
    return mu, std, gp


def display_gp_mean_uncertainty(kernel, optimizer, pbounds,BadIter):
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


def bayesian_optimisation(kernel, optimizer, utility, init_points, n_iter, pbounds, nbrs,RandomSeed,ModelEmbedding,BCT_models,BCT_Run,Sparsities_Run,Data_Run,Ages,CommunityIDs,data1,data2,ClassOrRegress,MultivariateUnivariate):

    BadIters=np.empty(0)





    LastModel=-1
    

    Iter=0
    while Iter < init_points + n_iter:
        np.random.seed(RandomSeed+Iter)
        # If burnin
        if Iter < init_points:
            # Choose point in space to probe next in search space randomly
            next_point_to_probe = {'b1': np.random.uniform(pbounds['b1'][0],
                                                           pbounds['b1'][1]),
                                   'b2': np.random.uniform(pbounds['b2'][0],
                                                           pbounds['b2'][1])}
            print("Next point to probe is:", next_point_to_probe)
            s1, s2 = next_point_to_probe.values()

        # if optimization
        else:
            # Choose point in space to probe next in search space using optimizer
            next_point_to_probe = optimizer.suggest(utility)
            print("Next point to probe is:", next_point_to_probe)
            s1, s2 = next_point_to_probe.values()

        # convert suggested coordinates to np array
        Model_coord = np.array([[s1, s2]])
        # find the index of the models that are closest to this point
        distances, indices = nbrs.kneighbors(Model_coord)

        # I order to reduce repeatedly sampling the same point, check if suggested point was sampled last and then check in ModelNums what the name/index
        # of that model is, if was recently sampled then take the second nearest point. 
        if LastModel == np.asscalar(indices[0][0]):
            TempModelNum = np.asscalar(indices[0][1])
            ActualLocation = ModelEmbedding[np.asscalar(indices[0][1])]
            Distance=distances[0][1]
        else:
            TempModelNum = np.asscalar(indices[0][0])
            ActualLocation = ModelEmbedding[np.asscalar(indices[0][0])]
            Distance=distances[0][0]
        

       
        if (Distance <10 or Iter<init_points):   #Hack: because space is continuous but analysis approaches aren't, we penalize points that are far (>10 distance in model space) from any actual analysis approaches by assigning them the value of the worst performing approach in the burn-in
            LastModel = TempModelNum
            BadIters=np.append(BadIters,0)
        # Call the objective function and evaluate the model/pipeline
            if MultivariateUnivariate<0:
                target=objectiveFunc(TempModelNum,Ages,Sparsities_Run,Data_Run,BCT_models,BCT_Run,CommunityIDs,data1,data2,ClassOrRegress)
                print("Next Iteration")
                print(Iter)
                # print("Model Num %d " % TempModelNum)
                print('Print indices: %d  %d' % (indices[0][0], indices[0][1]))
                print(Distance)
                print("Target Function: %.4f" % (target))
                print(' ')
                seed(Iter)
                # This is a hack. Add a very small random number to the coordinates so
                # that even if the model has been previously selected the GP thinks its
                # a different point, since this was causing it to crash
                TempLoc1 = ActualLocation[0] + (np.random.random_sample(1) - 0.5)/10
                TempLoc2 = ActualLocation[1] + (np.random.random_sample(1) - 0.5)/10
            else:
                target=objectiveFuncUnivariate(TempModelNum,Ages,Sparsities_Run,Data_Run,BCT_models,BCT_Run,CommunityIDs,data1,data2,MultivariateUnivariate)
                print("Next Iteration")
                print(Iter)
                
                print('Print indices: %d  %d' % (indices[0][0], indices[0][1]))
                print(Distance)
                print("Target Function: %.4f" % (target))
                print(' ')
                
                seed(Iter)
                # This is a hack to add very small random number to the coordinates so
                # that even if the model has been previously selected the GP thinks its
                # a different point, since this was causing it to crash
                TempLoc1 = ActualLocation[0] + (np.random.random_sample(1) - 0.5)/10
                TempLoc2 = ActualLocation[1] + (np.random.random_sample(1) - 0.5)/10

        else: 
            
            newlist = sorted(optimizer.res, key=lambda k: k['target'])
            target=newlist[0]['target']
            LastModel = -1
            
            print("Next Iteration")
            print(Iter)
            # print("Model Num %d " % TempModelNum)
#            print('Print indices: %d  %d' % (indices[0][0], indices[0][1]))
            print(Distance)
            print("Target Function Default Bad: %.4f" % (target))
            BadIters=np.append(BadIters,1)
            
            print(' ')
            seed(Iter)
            TempLoc1 = Model_coord[0][0]
            TempLoc2 = Model_coord[0][1]
            n_iter=n_iter+1
    
        Iter=Iter+1

        # Update the GP data with the new coordinates and model performance
        register_sample = {'b1': TempLoc1, 'b2': TempLoc2}
        optimizer.register(params=register_sample, target=target)
    return BadIters


# TODO: Check if ModelDict has the same pipeline as fitted_model
# Save the results in a pickle
