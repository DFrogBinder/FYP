import numpy as np
import os
import matplotlib.pyplot as plt 
import pydicom
import warnings
import png
import pandas as pd 
import numpy.ma as ma
import scipy.misc
from sklearn.linear_model import LinearRegression
from PIL import Image as im
from pydicom import dcmread
from pydicom.data import get_testdata_file
from scipy.optimize import curve_fit
from pathlib import Path
from time import process_time, sleep
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


def Model(x, S0,bv): 
        return  S0*np.exp(bv * (-x))
def Fit(px,bv):
        t1_start = process_time() 
                
        ItNum = 800 # Maximum Number of iterations for each point
        
        bnds = ([0], [np.inf])
        
        #Data =  loadmat('var')
        
        yDLin = px
        xDLin = bv
        
        yD = px
        sz = yD.shape
        yD = np.reshape(yD,[sz[0]*sz[1], sz[2]],order='F')
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
        
        # generate initial parameter values
        P1,P2 = LinFit(yDLin,xDLin)
        Error = []
        Rows = []
        ##########################################################################################
        # Iterating through the matrix row by row
        for row in yD.T: 
                if c == yD.shape[1]-1:
                        
                                # curve fit the test data
                                p1 = P1.T[c]
                                p2 = P2[c]

                                RowFit,pocv = curve_fit(func, xD,row, p0=[p1,p1],maxfev=ItNum,
                                                        method='lm',absolute_sigma='True'
                                                        ,jac=grad)              
                                S0.append(RowFit[1])
                                adc.append(RowFit[0]) 
                                Rows.append(RowFit)  
                                modelPredictions = func(xD, *RowFit) 

                                absError = modelPredictions - row
                                '''
                                SE = np.square(absError) # squared errors
                                MSE = np.mean(SE) # mean squared errors
                                RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
                                Rsquared = 1.0 - (np.var(absError) / np.var(row))
                                Error.append(Rsquared)
                                #print('\n Row ',c,'out of ',yD.shape[1])
                                #print('\n Parameters: ', RowFit)
                                #print('\n Rsquared',Rsquared)
                                #print('\n Sleep')  
                                '''                        
                                c += 1
                                break
                else:                        
                                # curve fit the test data
                                p1 = P1.T[c]
                                p2 = P2[c]

                                RowFit,pocv = curve_fit(func,xD,row, p0=[p1,p1],maxfev=ItNum,
                                                        method='lm',absolute_sigma='True',jac=grad
                                                    )              
                                S0.append(RowFit[1])
                                adc.append(RowFit[0]) 
                                Rows.append(RowFit)  
                                modelPredictions = func(xD, *RowFit) 

                                absError = modelPredictions - row
                                '''
                                SE = np.square(absError) # squared errors
                                MSE = np.mean(SE) # mean squared errors
                                RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
                                Rsquared = 1.0 - (np.var(absError) / np.var(row))
                                Error.append(Rsquared)
                                #print('\n Row ',c,'out of ',yD.shape[1])
                                #print('\n Parameters: ', RowFit)
                                #print('\n Rsquared',Rsquared)    
                                '''                    
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
        ''''
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
        '''
        return S0,adc
def LinFit(DiffImage, bv):
                sz = DiffImage.shape  
                
                bvalues = bv
                S = DiffImage
                # Flattens the image into a vector
                S = np.reshape(DiffImage,[sz[0]*sz[1], sz[2]],order='F')
                S[S == 0] = 1e-16  # Removes zeroes to prevent erros
                
                S[S==0]=np.amin(S[S>0])
                
                logS = np.log(S)
                logS = logS.T

                bv = np.reshape(bv,[sz[2],1])
                logS0_ADC = np.linalg.lstsq(bv,logS,rcond=None)[0].T
                S0 = np.linalg.lstsq(bv,logS,rcond=None)[1]

                logS0_ADC = np.asarray(logS0_ADC).T
                S0 = np.asarray(S0).T
                where_are_NaNs = np.isnan(logS0_ADC)
                logS0_ADC[where_are_NaNs] = 0
                return logS0_ADC,S0

def Scratch():
        PathDicom = "D:\IDL\PatientData\DICOM"
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
        Bvalues=[]
        locationMatrix = []
        Data=[]
        SortedImages = {}
        SortedBvals = {}
        #loop through all the DICOM files
        print("Loading Data...")
        for filenameDCM in tqdm(lstFilesDCM):
                # read the file
                ds = pydicom.read_file(filenameDCM)
                Data.append(ds)
                locationMatrix.append(ds.SliceLocation)
                ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
            
        locationMatrix = np.asarray(locationMatrix) 
        locationMatrix = np.unique(locationMatrix)
        
        print("Sorting Images")
        for index,location in zip(tqdm(range(len(locationMatrix)-1)),locationMatrix):
                key = "Position_"+str(index)
                SortedImages[key]=[]
                SortedBvals[key]=[]
                for image in Data:
                        if float(image.SliceLocation)== location:
                                SortedImages[key].append(image.pixel_array)
                                SortedBvals[key].append(int(image[0x0019,0x100c].value))
                
        Fitted_Images =[]
        NL_Fitted_Images=[]
        # Linear fittet
        print("Performing Liner fit")
        for image in tqdm(SortedImages):
                ImageMatrix = np.transpose(np.asarray(SortedImages[image]))
                bv = np.asarray(SortedBvals[image])
                logS0_ADC,S0 = LinFit(ImageMatrix,bv)
                logS0_ADC = np.reshape(logS0_ADC,[172,172])
                Fitted_Images.append(logS0_ADC)
        Fitted_Images = np.asarray(Fitted_Images)

        print("Performing Non-Liner fit")
        for image in tqdm(SortedImages):
                ImageMatrix = np.transpose(np.asarray(SortedImages[image]))
                bv = np.asarray(SortedBvals[image])
                NL_S0,NL_logS0_ADC = Fit(ImageMatrix,bv)
                NL_logS0_ADC = np.reshape(NL_logS0_ADC,[172,172])
                NL_Fitted_Images.append(NL_logS0_ADC)
        NL_Fitted_Images = np.asarray(NL_Fitted_Images)

        # Code to create .gif file of the fitted images
        print("Creting GIF image...")
        fig, ax = plt.subplots(figsize=(5, 8))
        def update(i):
                im_normed = NL_Fitted_Images[i,:,:]
                ax.imshow(im_normed,cmap='gray')
                ax.set_axis_off()
                print(i)
        anim = FuncAnimation(fig, update, frames=np.arange(0, 29), interval=1).save("Anim.gif")
        plt.close()
        
        # Non-linear Fitter
        #Fit(FirstImage,bv)
        
Scratch()