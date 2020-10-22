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

    os.mkdir(path)

    # Load Data and Export

    

fit_measured_2d()