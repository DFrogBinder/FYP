import numpy as np
import os
import functions 


def fit_measured_2d(DataSet = 'RIFF_C03',   # Default values
                    tolerance = 0.1, 
                    grid_size = 1, 
                    NOMOCO = True):

    Decomposed = 0
    Bypass_Translation = 0

    # Display settings

    Parameters = ['FP', 'TP', 'PS', 'TE']
    Range = np.zeros((4,2))
    #TODO:Implement the rest of the plotting settings

    # Export Path

    folder = DataSet
    if tolerance != 0:
        folder = folder + '_' + str(tolerance)
    if grid_size != 0:
        folder = folder + '_' + str(grid_size)
    
    path = os.path.join(functions.MDR_path('Results'),folder)
    if os.path.isdir(path)==True:
        print("Result folder already exist")
    else:
        os.mkdir(path)


    # Load Data and Export
    time,Ca,Source,Baseline = functions.Load_Data(DataSet)
    n = Source.shape
    Independent = {"t":time, "ca":Ca, "n0":Baseline}
    File = os.path.join(path,'Dynamics')
    functions.Export_Gif(Source,File,[0,np.amax(Source)])

    return
    

fit_measured_2d()