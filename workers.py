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
U=1
K=1
J=1
epsln=0.001
hyper_params=np.array([a_alpha,b_alpha,a_beta,b_beta,a_betaf,b_betaf,a_gama,b_gama,a_teta0,b_teta0,a_teta1,b_teta1])

if os.path.exists('G.npy'):
    G = np.load('G.npy')
if os.path.exists('YF.npy'):
    YF = np.load('YF.npy')
if os.path.exists('F.npy'):
    F = np.load('F.npy')
if os.path.exists('YF_missing.npy'):
    YF_missing = np.load('YF_missing.npy')
#if os.path.exists('YF_missing1.npy'):
    #YF_missing1 = np.load('YF_missing1.npy')
    #print(YF_missing1)
if os.path.exists('X.npy'):
    X = np.load('X.npy')
n,T=X.shape[0],X.shape[1]
unique_rows = np.unique(F.T, axis=0)
#calculate y_missing:
Y=np.zeros((n,T))
for i in range(n):
    for j in range(unique_rows.shape[0]):
        if unique_rows[j,i]==1:
            Y[i,:]=YF_missing[j,:]
def algrthm(params,X):
    
    prob=[]
    for i in range(U):
        
        param=[]
        param.append(params)
        cal=Code.Calculate_X(K,T,X,G,F,Y,params,P)
        X=cal[0]
        pos_probs=cal[1]
        R=Code.R_(G,X,params,F)
        if (i!=U-1):
            prm=Code.Params(R,G,F,X,n,T,Y,hyper_params)
            params=prm[0]
            R=prm[1]
            if i>1 & Code.epsilone(param[-1],params):
                param.append(params)
        prob.append(pos_probs)
        np.savetxt("MyF.txt",pos_probs)
    return X,prob,param