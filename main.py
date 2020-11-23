import numpy as np
import os
import matplotlib.pyplot as plt 
import pydicom
import warnings
import png
import pandas as pd 
import numpy.ma as ma
from PIL import Image as im
import scipy.misc
from pydicom import dcmread
from pydicom.data import get_testdata_file
from scipy.optimize import curve_fit
from pathlib import Path
from time import process_time, sleep
from matplotlib.animation import FuncAnimation


def Model(x, S0): 
        return  S0*np.exp(-1000 * x)
def Fit(px,bv):
        t1_start = process_time() 
                
        ItNum = 800 # Maximum Number of iterations for each point
        
        bnds = ([0], [np.inf])
        
        #Data =  loadmat('var')
        
        yDLin = px
        xDLin = bv
        
        yD = px
        sz = yD.shape
        yD = np.reshape(yD,[sz[0]*sz[1], 1],order='F')
        yD = yD.tolist()
        yD = np.asarray(yD)
        yD = yD.astype(np.float64)
        yD = yD.T

        
        xD = bv
        xD = xD.tolist()
        xD = np.asarray(xD)
        xD = np.reshape(xD,[len(xD.T)],order='F')
        xD = xD.astype(np.float64)
        xD = xD.T
        
        S0 = []
        adc = []
        c = 0
        
        ##########################################################################################
        # Functions section 
        def func(x, bv, S0): 
                return  S0*np.exp(bv * (-x))
        
        def grad(t,a,b):
                d1 = np.exp(a * (-t))
                d2 = -np.exp(a*(-t))*t*b
                return np.array([d2,d1]).T

        # function for genetic algorithm to minimize (sum of squared error)
        def sumOfSquaredError(parameterTuple):
                warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
                val = func(xD, *parameterTuple)   
                return np.sum((row - val) ** 2.0)
        
        # Linear fitting for initial values 
        def LinFit(DiffImage, bv):
                sz = DiffImage.shape  
                
                bvalues = bv
                S = DiffImage
        
                S = np.reshape(DiffImage,[sz[0]*sz[1], 1],order='F')
                
                S[S == 0] = 1e-16
                
                S[S==0]=np.amin(S[S>0])
                
                logS = np.log(S)
                logS = logS.T

                B = np.vstack((np.ones(bvalues.shape),bvalues))
                B = B.T
                
                logS0_ADC = np.linalg.lstsq(B,logS,rcond=None)[0].T
        
                where_are_NaNs = np.isnan(logS0_ADC)
                logS0_ADC[where_are_NaNs] = 0
                return logS0_ADC
        
        # generate initial parameter values
        geneticParameters = LinFit(yDLin,xDLin)
        Error = []
        Rows = []
        ##########################################################################################
        # Iterating through the matrix row by row
        for row in yD:
            print(row)
        for row in yD.T: 
                if c == yD.shape[1]-1:
                        
                                # curve fit the test data
                                Guess = geneticParameters[c]
                                p1 = Guess[0]
                                p2 = Guess[1]
                                RowFit,pocv = curve_fit(func, xD, row, p0=[p2,p1],maxfev=ItNum,
                                                        method='lm',absolute_sigma='True'
                                                        ,jac=grad)              
                                S0.append(RowFit[1])
                                adc.append(RowFit[0]) 
                                Rows.append(RowFit)  
                                modelPredictions = func(xD, *RowFit) 

                                absError = modelPredictions - row

                                SE = np.square(absError) # squared errors
                                MSE = np.mean(SE) # mean squared errors
                                RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
                                Rsquared = 1.0 - (np.var(absError) / np.var(row))
                                Error.append(Rsquared)
                                print('\n Row ',c,'out of ',yD.shape[1])
                                #print('\n Parameters: ', RowFit)
                                print('\n Rsquared',Rsquared)
                                print('\n Sleep')                          
                                c += 1
                                break
                else:                        
                                # curve fit the test data
                                Guess = geneticParameters[c]
                                p1 = Guess[0]
                                p2 = Guess[1]
                                RowFit,pocv = curve_fit(func,p2, row, p0=[xD,p1],maxfev=ItNum,
                                                        method='lm',absolute_sigma='True'
                                                    )              
                                S0.append(RowFit[1])
                                adc.append(RowFit[0]) 
                                Rows.append(RowFit)  
                                modelPredictions = func(xD, *RowFit) 

                                absError = modelPredictions - row

                                SE = np.square(absError) # squared errors
                                MSE = np.mean(SE) # mean squared errors
                                RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
                                Rsquared = 1.0 - (np.var(absError) / np.var(row))
                                Error.append(Rsquared)
                                print('\n Row ',c,'out of ',yD.shape[1])
                                #print('\n Parameters: ', RowFit)
                                print('\n Rsquared',Rsquared)                        
                                c += 1
                
                
        t1_stop = process_time()         
        Error = np.asarray(Error)
        Sum = np.sum(Error)
        Mean = Sum/yD.shape[1]
        print('\n The mean Rsquared is: ', Mean)
        print("Elapsed time during the whole program in seconds:", 
                                         t1_stop-t1_start)
        S0 = np.asarray(S0)
        S0 = np.reshape(S0,[yDLin.shape[0], yDLin.shape[1]], order='F')
        adc = np.asarray(adc)
        adc = np.reshape(adc,[yDLin.shape[0], yDLin.shape[1]], order='F')
        ##########################################################################################

        # Graphics output section
        
        def ModelAndScatterPlot(graphWidth, graphHeight):
                f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
                axes = f.add_subplot(111)

        # first the raw data as a scatter plot
                axes.plot(xD, yD,  'r+')

        # create data for the fitted equation plot
                xModel = np.linspace(min(xD), max(xD))
                yModel = func(xModel, *RowFit)

        # now the model as a line plot
                axes.plot(xModel, yModel)

                axes.set_xlabel('bvalues') # X axis data label
                axes.set_ylabel('Y Data') # Y axis data label

                plt.show()
                plt.close('all') # clean up after using pyplot

        graphWidth = 800
        graphHeight = 600
        ModelAndScatterPlot(graphWidth, graphHeight)
        
        xModel = np.linspace(min(xD), max(xD))
        yModel = func(xModel, *RowFit)
        
        return S0,adc

def LinFit(DiffImage, bv):
                sz = DiffImage.shape  
                
                bvalues = bv
                S = DiffImage
        
                S = np.reshape(DiffImage,[sz[0]*sz[1], 1],order='F')
                S[S == 0] = 1e-16
                
                S[S==0]=np.amin(S[S>0])
                
                logS = np.log(S)
                logS = logS.T

                #B = np.vstack((np.ones(bvalues.shape),bvalues))
                #B = B.T
                #bv = bv.reshape(bv,[1,1])
                #logS = logS.reshape(logS,[logS.shape[0],1],order='F')

                logS0_ADC = np.linalg.lstsq(bv,logS,rcond=None)[0].T
        
                where_are_NaNs = np.isnan(logS0_ADC)
                logS0_ADC[where_are_NaNs] = 0
                logS0_ADC = np.reshape(logS0_ADC,[128,128],order='F')
                return logS0_ADC

def Scratch():
    PathDicom = "./Data/"
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    # Get ref file
    RefDs = pydicom.read_file(lstFilesDCM[0])

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    #loop through all the DICOM files
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = pydicom.read_file(filenameDCM)
        # store the raw image data
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    FirstImage = ArrayDicom[:,:,0]
    bv = np.full((1,1),1000)
    
    LinFitImageList = np.empty([128,128])
    # Linear fittet
    
    for image in range(len(ArrayDicom[0,0,:])):
        logS0_ADC = LinFit(ArrayDicom[:,:,image],bv)
        LinFitImageList = np.dstack((LinFitImageList,logS0_ADC)) 
    
    logS0_ADC = LinFit(FirstImage,bv)
    
    fig, ax = plt.subplots(figsize=(5, 8))
    def update(i):
        im_normed = LinFitImageList[:,:,i]
        ax.imshow(im_normed,cmap='gray')
        ax.set_axis_off()
        print(i)
    anim = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=1).save("Anim.gif")
    plt.close()
   
    # Non-linear Fitter
    #Fit(FirstImage,bv)
    

    plt.figure()
    plt.imshow(LinFitImageList[:,:,1],cmap="gray")
    plt.figure()
    plt.imshow(ArrayDicom[:,:,0],cmap="gray")
    plt.show()
    plt.close()

Scratch()