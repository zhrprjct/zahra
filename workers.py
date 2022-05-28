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
import Code
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
U=2
K=2
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
    X = np.load('Xreal.npy')
n,T=X.shape[0],X.shape[1]

unique_rows = np.unique(F, axis=0)
#calculate y_missing:
Y=np.zeros((n,T))
for i in range(n):
    for j in range(unique_rows.shape[0]):
        if unique_rows[j,i]==1:
            Y[i,:]=YF[j,:]
def algrthm(params,X):
    prob=[]
    for i in range(U):
        np.savetxt('iteration.txt' ,[i,U])
        param=[]
        param.append(params)
        cal=Code.Calculate_X(K,T,X,G,F,Y,params,P)
        X=cal[0]
        pos_probs=cal[1]
        R=Code.R_(G,X,params,F)
        if (i!=U-1):
            prm=Code.Params(R,G,F,X,n,T,YF,hyper_params)
            params=prm[0]
            R=prm[1]
            if i>1 & Code.epsilone(param[-1],params):
                param.append(params)
        prob.append(pos_probs)
    return X,prob,param