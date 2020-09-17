import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def Normalize(x):
    mu = np.mean(x[:,:],0).T
    sigma = np.std(x[:,:],0).T
    x[:,:] = (x[:,:]-mu)/sigma   
    return x

def Cost(x,y,theta):
    j = np.sum(np.square((x.dot(theta)- y)))/(2*len(y))
    return j

def GradientDescent(x,y,theta,alpha,n):
    j_hist = np.zeros((n,1))
    for i in range(n):        
        theta = theta-(alpha*(x.dot(theta)-y).T.dot(x)/len(y)).T
        j_hist[i] = Cost(x,y,theta)
    return theta, j_hist

def NormalEquation(x,y):
    return ((np.linalg.inv((x.T).dot(x))).dot(x.T)).dot(y)
