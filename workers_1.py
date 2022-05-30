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
import Code_1
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
        cal=Code_1.Calculate_X(K,T,X,G,F,Y,params,P)
        X=cal[0]
        pos_probs=cal[1]
        prob.append(pos_probs)
    return X,prob
# functions for multiprocessing
def Step_Gibbs(X,G,Y):
    Trained=[]
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
