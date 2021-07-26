import re
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from scipy.signal import savgol_filter


def Extract_Loss(path):
    Extracted_Data = []
    for line in open(path):
        if 'loss' in line:
            StrSplit = line.split()
            for i,k in zip(StrSplit,range(len(StrSplit))):
                if i == 'loss:':
                    Extracted_Data.append(float(StrSplit[k+1]))
    
    Extracted_Data = np.asarray(Extracted_Data)
    kernel_size = 100
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(Extracted_Data, kernel, mode='same')   
    X_data = range(len(Extracted_Data))      
    
    fig,ax=plt.subplots()       
    ax.plot(X_data,Extracted_Data)
    
    fig,ax=plt.subplots()       
    ax.plot(X_data,data_convolved)
    plt.show()

Extract_Loss('/Users/boyanivanov/log/Log.txt')