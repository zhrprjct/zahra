

import numpy as np
import random
import sympy as sym
from datetime import datetime
from scipy.stats import beta
from sklearn.metrics import accuracy_score
import winsound
import os
epsln=0.001

# function to count the number of the infected neighbores of i at t:
def CNbr(G,X):
    n,T=X.shape[0],X.shape[1]
    C=np.zeros((T,n))
    for t in range(T):
        C[t]=G[t].dot(X.T[t])
    return C.T

def transition(X,t,G,F,j,param):
    alpha_=param[0]
    beta_=param[1]
    betaf=param[2]
    gama_=param[3]
    c=CNbr(G,X)[j][t]
    number_of_infected_members_in_family=F.dot(X.T[t])[j]
    k=X[j,t]-2*X[j,t+1]
    if k==0:
        return 1-alpha_-beta_*c-betaf*number_of_infected_members_in_family
    elif k==-2:
        return alpha_+beta_*c+betaf*number_of_infected_members_in_family
    elif k==1:
        return gama_
    else:
        return 1-gama_
    
def Sample_hidden_state(pos_probs,X,G,F,Y,param,P,t):
    unique_rows = np.unique(F, axis=0)
    alpha_=param[0]
    beta_=param[1]
    betaf=param[2]
    gama_=param[3]
    theta_0_=param[4]
    theta_1_=param[5]
    n,T=X.shape[0],X.shape[1]

    for i in range(n):
        if t==0:
            p_0,p_1=P,1-P
        else:
            p_0,p_1=1,1
        pow0=np.count_nonzero(Y[i,t]==0)
        pow1=np.count_nonzero(Y[i,t]==1)
        #pow1_=np.count_nonzero(Y[i,t]==-1)
        number_of_members_in_family=np.sum(unique_rows[family_index(i,unique_rows)])
        
        X[i,t]=0
        number_of_infected_members_in_family0=F.dot(X.T[t])[i]
        number_of_healthy_members_in_family0=number_of_members_in_family-number_of_infected_members_in_family0
        p_0=(1/number_of_members_in_family)*p_0*((1-theta_0_)*number_of_healthy_members_in_family0+(1-theta_1_)*number_of_infected_members_in_family0)**pow0*(theta_1_*number_of_infected_members_in_family0+theta_0_*number_of_healthy_members_in_family0)**pow1
        if (t==0):
            c=G[t].dot(X.T[t])[i]
        else:    
            c=G[t-1].dot(X.T[t-1])[i]
    
        if t!=0:
            if X[i,t-1]==0:
                p_0=p_0*(1-alpha_-beta_*c-betaf*number_of_infected_members_in_family0)
            else:
                p_0=p_0*gama_
        
        X[i,t]=1
        number_of_infected_members_in_family1=F.dot(X.T[t])[i]
        number_of_healthy_members_in_family1=number_of_members_in_family-number_of_infected_members_in_family1
        p_1=(1/number_of_members_in_family)*p_1*((1-theta_0_)*number_of_healthy_members_in_family1+(1-theta_1_)*number_of_infected_members_in_family1)**pow0*(theta_1_*number_of_infected_members_in_family1+theta_0_*number_of_healthy_members_in_family1)**pow1

        if (t==0):
            c=G[t].dot(X.T[t])[i]
        else:    
            c=G[t-1].dot(X.T[t-1])[i]
        if t!=0:
            if X[i,t-1]==0:
                p_1=p_1*(alpha_+beta_*c+betaf*number_of_infected_members_in_family1)
            else:
                p_1=p_1*(1-gama_)
        family_members=unique_rows[family_index(i,unique_rows)]
        
        if t!=T-1:        
            X[i,t]=0
            for j in np.where(family_members==1)[0]:
                if j!=i:
                    p_0=p_0*transition(X,t,G,F,j,param)
            for j in np.where(G[t][i]==1)[0]:
                p_0=p_0*transition(X,t,G,F,j,param)
            X[i,t]=1
            for j in np.where(family_members==1)[0]:
                if j!=i:
                    p_1=p_1*transition(X,t,G,F,j,param)
            for j in np.where(G[t][i]==1)[0]:
                p_1=p_1*transition(X,t,G,F,j,param)
        if t==T-1:
            if X[i,t-1]==0:
                X[i,t]=0
                c=G[t].dot(X.T[t])[i]
                number_of_infected_members_in_family=F.dot(X.T[t])[i]
                p_0=p_0*(1-alpha_-beta_*c-betaf*number_of_infected_members_in_family)
                X[i,t]=1
                c=G[t].dot(X.T[t])[i]
                number_of_infected_members_in_family=F.dot(X.T[t])[i]
                p_1=p_1*(alpha_+beta_*c+betaf*number_of_infected_members_in_family)
            else:
                p_0=p_0*gama_
                p_1=p_1*(1-gama_)
        if p_0+p_1==0:            
            l=0.5
        else:
            l=p_1/(p_0+p_1)
        if (l<0)|(l>1): 
            print(p_0,p_1)
        X[i,t]=np.random.binomial( 1,  l,size=None)    
        pos_probs[i,t]=l
    return X ,pos_probs   


# In[6]:


# Gibbs sampling to obtain X, as new sample of posterior distribution:
def Calculate_X(K,T,X,G,F,Y,param,P):
    n,T=X.shape[0],X.shape[1]
    pos_probs=np.zeros((n,T))
    for k in range(K):
        for t in range(T):
            np.savetxt('my_file.txt' ,[k,t])

            hidden_states=Sample_hidden_state(pos_probs,X,G,F,Y,param,P,t)
            X=hidden_states[0]
            pos_probs=hidden_states[1]
    
    return X  ,pos_probs              

def Accuracy_(X,X_):
    Xflat = np.hstack(np.hstack(X))
    X_flat = np.hstack(np.hstack(X_))
    result=accuracy_score(Xflat, X_flat,normalize=True)
    return result

# function to define auxiliary variable R_(n,t):
def R_(G,X,params,F):
    alpha_,beta_,betaf,gama_,theta_0_,theta_1_=params[0],params[1],params[2],params[3],params[4],params[5]
    n,T=X.shape[0],X.shape[1]
    infected_neighbore=np.array(CNbr(G,X))
    R=np.zeros((n,T))+1
    for i in range(n):
        for t in range(T-1):
            c=int(infected_neighbore[i,t])
            cf=int(F.dot(X.T[t])[i])
            pr_a=alpha_/(alpha_+beta_*c+betaf*cf)
            pr_b=beta_/(alpha_+beta_*c+betaf*cf)
            pr_bf=betaf/(alpha_+beta_*c+betaf*cf)
            v=np.random.multinomial(1, [pr_a]+[pr_b]*c+[pr_bf]*cf)
            if (X[i][t]==0)&(X[i][t+1]==1):
                if v[0]==1:
                    R[i,t]=0
                elif (cf!=0):
                    if ((v[-cf:]==1).any()):
                        R[i,t]=3
                else:    
                    R[i,t]=2
    return R

# function to sample new parameters and update parameters:
def Params(R,G,F,X,n,T,YF,hyper_params):
    
    a_alpha=hyper_params[0]
    b_alpha=hyper_params[1]
    a_beta=hyper_params[2]
    b_beta=hyper_params[3]
    a_betaf=hyper_params[4]
    b_betaf=hyper_params[5]
    a_gama=hyper_params[6]
    b_gama=hyper_params[7]
    a_teta0=hyper_params[8]
    b_teta0=hyper_params[9]
    a_teta1=hyper_params[10]
    b_teta1=hyper_params[11]
    unique_rows=np.unique(F,axis=0)  
    TP=np.sum(np.multiply(unique_rows.dot(X),YF_missing1))
    FP=np.count_nonzero(unique_rows.dot(X)-YF_missing1==-1)
    
    infected_neighbore=np.array(CNbr(G,X))
    a_alpha, b_alpha=a_alpha +  np.count_nonzero(R==0) , b_alpha +np.count_nonzero(X==0)- np.count_nonzero(R==0)
    alpha_=Sample_alpha(a_alpha, b_alpha)
    a_beta,b_beta=a_beta + np.count_nonzero(R==2) , b_beta +np.sum(np.multiply((1-X),infected_neighbore))-np.count_nonzero(R==2)
    beta_=Sample_beta(a_beta,b_beta)
    a_betaf ,b_betaf=a_betaf + np.count_nonzero(R==3) , b_betaf +np.sum(np.multiply((1-X),F.dot(X)))-np.count_nonzero(R==3)
    betaf=Sample_betaf(a_betaf ,b_betaf)
    while alpha_>beta_:
        np.savetxt("myfil.txt",[alpha_,beta_])
        alpha_=Sample_alpha(a_alpha, b_alpha)
        beta_=Sample_beta(a_beta, b_beta)
    while beta_>betaf:
        np.savetxt("myfil.txt",[beta_,betaf])
        betaf=Sample_betaf(a_betaf, b_betaf)
    gama_=Sample_gama(a_gama +np.count_nonzero((X[:,:-1]-X[:,1:])==1), b_gama+np.sum(X)-np.count_nonzero((X[:,:-1]-X[:,1:])==1))
    theta_0_=Sample_theta0( a_teta0+FP,b_teta0+np.count_nonzero((unique_rows.dot(X))==0)-FP)
    theta_1_=Sample_theta1( a_teta1+TP,b_teta1+np.sum(unique_rows.dot(X))-TP)
    params=np.array([alpha_,beta_,betaf,gama_,theta_0_,theta_1_])
    R=R_(G,X,params,F)
    param=[alpha_,beta_,betaf,gama_,theta_0_,theta_1_]
    return param,R


# In[10]:


def Sample_alpha(a_alpha, b_alpha):
    for i in beta.rvs(a_alpha, b_alpha, size=10000):
        if (i>0.001)&(i<0.2):
            alpha_=round(i,3)
            break
    return alpha_        

def Sample_beta(a_beta, b_beta):
    for i in beta.rvs(a_beta, b_beta, size=10000):
        if (i>0.001)&(i<0.0451):
            beta_=round(i,4)
            break
    return beta_        

def Sample_betaf(a_betaf, b_betaf):
    for i in beta.rvs(a_betaf, b_betaf, size=1000):
        if (i>0.002)&(i<0.5):
            betaf=round(i,4)
            break
    return betaf        

def Sample_gama(a_gama,b_gama):
    for i in beta.rvs(a_gama, b_gama, size=10000):
        if (i>0.1)&(i<0.5):
            gama_=round(i,3)
            break
    return gama_  

def Sample_theta0(a_teta0, b_teta0):
    for i in beta.rvs(a_teta0, b_teta0, size=10000):
        if (i>0.01)&(i<0.51):
            theta_0_=round(i,3)
            break
    return theta_0_  

def Sample_theta1(a_teta1, b_teta1):
    for i in beta.rvs(a_teta1, b_teta1, size=10000):
        if 0.990>i>0.78:
            theta_1_=round(i,3)
            break
    return theta_1_  

def add_noise(YF,noise_percent,unique_rows):
    
    number_of_families=unique_rows.shape[0]
    indx=random.sample(range(number_of_families), number_of_families)
    tndx=random.sample(range(T), int(noise_percent*T))
    for i in indx:
        for t in tndx:
            YF[i,t]=(YF[i,t]+1)*(1-YF[i,t])
    return YF        

# funtion to retun related family index of individual i:
def family_index(i,unique_rows):
    n=unique_rows.shape[1]
    for j in range(n):
        if unique_rows[j,i]==1:
            return j

def epsilone(a,b):
    return np.abs(a-b).all()>epsln

def estimate_Y(Family,X,params):
    theta_0=params[4]
    theta_1=params[5]
    unique_rows = np.unique(Family, axis=0)
    nf=unique_rows.shape[0]
    YF=np.zeros((nf,T))
    for t in range(T):
        for i in range(nf):
            number_of_members_in_family=np.sum(unique_rows[i])
            number_of_infected_members_in_family=unique_rows[i].dot(X.T[t])
            number_of_healthy_members_in_family= number_of_members_in_family-number_of_infected_members_in_family
            py1=(theta_0_*number_of_healthy_members_in_family+theta_1_*number_of_infected_members_in_family)/ number_of_members_in_family
            py0=((1-theta_0_)*number_of_healthy_members_in_family+(1-theta_1_)*number_of_infected_members_in_family)/ number_of_members_in_family
            l=py1/(py1+py0)
            YF[i,t]=np.random.binomial( 1, l,size=None)
    return YF

