import numpy as np
import sys
import math 
import scipy as sc
from scipy.io import loadmat
import matplotlib.pyplot as plt


def fit_ADC(px, bv):
    '''
    if isinstance(DiffImage,np.ndarray) == False:
       #messagebox.showinfo("Error", "The Diffusion image need to be an array")
        raise ValueError('The image needs to be an array!')
        
    if isinstance(bvalues,np.ndarray) == False:
        #messagebox.showinfo("Error", "The bvalues need to be an array")
        raise ValueError('bvalues needs to be an array!')
    
    Data =  loadmat('Tvar')
    '''
    
    DiffImage = px
    bvalues = bv
   
   
    sz = DiffImage.shape  
    
    S = np.reshape(DiffImage,[sz[0]*sz[1], sz[2]],order='F')

    S[S == 0] = 1e-16
    
    S[S==0]=np.amin(S[S>0])
    
    logS = np.log(S)
    
    logS = logS.T

    B = np.vstack((np.ones(bvalues.shape),bvalues))
    B = B.T
       
    logS0_ADC = np.linalg.lstsq(B,logS,rcond=None)[0].T
    
    S0 = np.reshape(np.exp(logS0_ADC[:,0]),[sz[0],sz[1]],order='F')
    adc = - np.reshape(logS0_ADC[:,1],[sz[0], sz[1]],order='F')
    
    where_are_NaNs = np.isnan(S0)
    S0[where_are_NaNs] = 0
    
    where_are_NaNs = np.isnan(adc)
    adc[where_are_NaNs] = 0
    
    return S0,adc
    
    





