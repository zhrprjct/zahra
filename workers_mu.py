print("hi im executing")
# import neccessary libraries
import numpy as np
import random
import sympy as sym
from datetime import datetime
from scipy.stats import beta
from sklearn.metrics import accuracy_score
import winsound
import os
import Code_mu
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

duration = 1000  # milliseconds
freq = 440  # Hz 
# initialize parameters for beta distributions:
a_alpha=1
b_alpha=400
a_beta=1
b_beta=800
a_betaf=8
b_betaf=800
a_gama=50
b_gama=200
a_teta0=10
b_teta0=1000
a_teta1=9000
b_teta1=300
P=1
U=1
K=1
J=1
n,T=100,365
epsln=0.001
hyper_params=np.array([a_alpha,b_alpha,a_beta,b_beta,a_betaf,b_betaf,a_gama,b_gama,a_teta0,b_teta0,a_teta1,b_teta1])

if os.path.exists('Greal.npy'):
    G = np.load('Greal.npy')
if os.path.exists('YFreal.npy'):
    YF = np.load('YFreal.npy')
if os.path.exists('Freal.npy'):
    F = np.load('Freal.npy')
if os.path.exists('YF_missingreal.npy'):
    YF_missing = np.load('YF_missingreal.npy')
if os.path.exists('YF_missing_1real.npy'):
    YF_missing1 = np.load('YF_missing_1real.npy')
if os.path.exists('Xreal.npy'):
    X_True = np.load('Xreal.npy')
n,T=X_True.shape[0],X_True.shape[1]
if os.path.exists('params.npy'):
    params = np.load('params.npy')
if os.path.exists('MissingData.npy'):
    params = np.load('MissingData.npy')    
MissingData= open('MissingData', 'r')
#function to plot figure 2:    
def plot_AUC_mu(MissingData)
    avg_roc=[]
    YF_missing=MissingData[0]
    mu=MissingData[1]
    unique_rows=np.unique(F)
    #calculate y_missing:
    Y=np.zeros((n,T))
    for i in range(n):
        for j in range(YF_missing.shape[0]):
            if unique_rows[j,i]==1:
                Y[i,:]=YF_missing[j,:]
    Trained=[]
    for time_step in range(10,360,10):
        G_=G[:time_step]
        Y_=Y[:,:time_step]
        X_=X[:,:time_step]
        #Train=Gibbs_train(hyper_params,G_,F,Y_,U,K,J)
        Train=algrthm(X_,G_,Y_)
        Trained.append(Train)
    roc_=[]
    for i in range(1,len(Trained)):
        Train=np.array(Trained[i][2][0])
        y_score=np.hstack(Train)
        y_test=np.hstack(X_True[:,:Train.shape[1]])
    
        roc_.append(plot_ROC(y_score,y_test))
    plt.plot(range(10,360,10),roc_)
    avg_roc=[mu,np.mean(np.array(roc_))]
    return avg_roc
# function to plot roc:
def plot_ROC(y_score,y_test):
    plt.figure()
    lw = 2
    #y_score=np.hstack(Train[2][j])
    #y_test=np.hstack(X)
    fpr, tpr, thresholds = roc_curve(y_test, y_score,pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC for family test result problem")
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

unique_rows = np.unique(F, axis=0)
#calculate y_missing:
Y=np.zeros((n,T))
for i in range(n):
    for j in range(unique_rows.shape[0]):
        if unique_rows[j,i]==1:
            Y[i,:]=YF[j,:]
        
# function for main algorithm
def algrthm(X,G,Y):
    prob=[]
    for i in range(U):
        np.savetxt('iteration.txt' ,[i,U])
        cal=Code_mu.Calculate_X(K,T,X,G,F,Y,params,P)
        X=cal[0]
        pos_probs=cal[1]
        prob.append(pos_probs)
    return X,prob
# functions for multiprocessing
def Step_Gibbs(arg):
    Trained=[]
    X=arg[0]
    G=arg[1]
    Y=arg[2]
    T=G.shape[0]
    for time_step in range(10,T,10):
        G_=G[:time_step]
        Y_=Y[:,:time_step]
        X_=X[:,:time_step]
        Train=algrthm(X_,G_,Y_)
        Trained.append(Train) 
    return Trained     
def Step_Gibbs_paralel(X):
    arg=[]
    pool_list=[]
    for i in range(1,5):
        time_step=i*90
        G_=G[:time_step]
        Y_=Y[:,:time_step]
        X_=X[:,:time_step]
        pool_list.append([X_,G_,Y_])
        
    if __name__ ==  '__main__': 
        with Pool(processes =4) as pool:

            parallel_output = pool.starmap(Step_Gibbs,pool_list )# use tqdm to show the progress
            pool.close()
            pool.join()
    Train=algrthm(X_,G_,Y_)
    return Train
