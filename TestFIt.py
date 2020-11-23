import sys
import scipy, matplotlib
import matplotlib.pyplot as plt
import warnings
import pandas as pd 
import numpy.ma as ma
from time import process_time, sleep
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from scipy.io import loadmat
np.set_printoptions(threshold=sys.maxsize)

def Fit():
        t1_start = process_time() 
                
        ItNum = 800 # Maximum Number of iterations for each point
        
        bnds = ([0], [np.inf])
        
        Data =  loadmat('Tvar')
        
        yDLin = Data['px']
        xDLin = Data['bv']
        
        yD = Data['px']
        sz = yD.shape
        yD = np.reshape(yD,[sz[0]*sz[1], sz[2]],order='F')
        yD = yD.tolist()
        yD = np.asarray(yD)
        yD = yD.astype(np.float64)
        yD = yD.T

        
        xD = Data['bv']
        xD = xD.tolist()
        xD = np.asarray(xD)
        xD = np.reshape(xD,[len(xD.T)],order='F')
        xD = xD.astype(np.float64)
        '''
        FitMask = mask
        fz = FitMask.shape
        FitMask = 1 - FitMask
        FitMask = FitMask.flatten(order='F')
        FitMaskC = FitMask
        
        for mask in range(9):
                FitMask = np.append(FitMask,FitMaskC)
        FitMask = np.reshape(FitMask,[sz[0]*sz[1], sz[2]],order='F')
        
        
        yDM = ma.array(yD,mask=FitMask,order='F')    
        yDM = pd.DataFrame(yDM)
        yDM = yDM.fillna(0)
        yDM = np.asarray(yDM)
        yDM = np.reshape(yDM,[sz[0]*sz[1], sz[2]],order='F')
        yDM = yDM.T
        '''
        
        S0 = []
        adc = []
        c = 0
        
        ##########################################################################################
        # Functions section 
        def func(x, a, b): 
                return  b*np.exp(a * (-x))
        
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
        
                S = np.reshape(DiffImage,[sz[0]*sz[1], sz[2]],order='F')
                
                S[S == 0] = 1e-16
                
                S[S==0]=np.amin(S[S>0])
                
                logS = np.log(S)
                logS = logS.T

                B = np.vstack((np.ones(bvalues.shape),bvalues))
                B = B.T
                
                logS0_ADC = np.linalg.lstsq(B,logS)[0].T
        
                where_are_NaNs = np.isnan(logS0_ADC)
                logS0_ADC[where_are_NaNs] = 0
                return logS0_ADC
        
        # generate initial parameter values
        geneticParameters = LinFit(yDLin,xDLin)
        Error = []
        Rows = []
        ##########################################################################################
        # Iterating through the matrix row by row
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
        '''
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
        '''
Fit()


    