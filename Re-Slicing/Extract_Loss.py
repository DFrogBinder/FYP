import re
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from math import factorial
from scipy.signal import savgol_filter


def Extract_Loss(path):
    NumberOfItters = 0
    Slice_Data = {}
    
    # Creates a list of each of the slices to later append the loss for each slice
    for i in range(5):
        Slice_Data[str(i)]=[]
    
    for line in open(path):
        if '1/22' in line:
            StrSplit = line.split()
            for i,k in zip(StrSplit,range(len(StrSplit))):
                if i == 'loss:':
                    Slice_Data['0'].append(float(StrSplit[k+1]))
        elif '2/25' in line:
            StrSplit = line.split()
            for i,k in zip(StrSplit,range(len(StrSplit))):
                if i == 'loss:':
                    Slice_Data['1'].append(float(StrSplit[k+1]))
        elif'3/25' in line:
            StrSplit = line.split()
            for i,k in zip(StrSplit,range(len(StrSplit))):
                if i == 'loss:':
                    Slice_Data['2'].append(float(StrSplit[k+1]))
        elif '4/25' in line:
            StrSplit = line.split()
            for i,k in zip(StrSplit,range(len(StrSplit))):
                if i == 'loss:':
                    Slice_Data['3'].append(float(StrSplit[k+1]))
        elif '5/25' in line:
            StrSplit = line.split()
            for i,k in zip(StrSplit,range(len(StrSplit))):
                if i == 'loss:':
                    Slice_Data['4'].append(float(StrSplit[k+1]))
    
    #Slice_Data['4'] = np.unique(Slice_Data['4']) # Filters repeating data
    for i in range(5):
        Slice_Data[str(i)] = np.asarray(Slice_Data[str(i)])
    
    # Data smoothing function
    '''[]
    kernel_size = 100
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(Extracted_Data, kernel, mode='same')   
    '''
    X_data = range(len(Slice_Data['0']))      
    X_data_sp = range(len(Slice_Data['3']))
    fig,ax=plt.subplots()       
    ax.plot(X_data,Slice_Data['0'])
    
    fig,ax=plt.subplots()       
    ax.plot(X_data_sp,Slice_Data['3'])
    plt.show()

Extract_Loss('/Users/boyanivanov/PresentationInfo/Log_Files/CT_Lung_Log.txt')
