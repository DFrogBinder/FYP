import numpy as np
import sys
import struct
import os
from collections import Counter
np.set_printoptions(threshold=sys.maxsize)


def MDR_path(which):
    if Counter("Data")==Counter(which):
        path = os.path.join(os.path.dirname( __file__ ),'Data')
    elif Counter("Results")==Counter(which):
        path = os.path.join(os.path.dirname( __file__ ),'Results')
    return(path)

def Load_Data(DataSet,time, Caif, Data, Baseline):
    if Counter('NonRigid_Motion')==Counter(DataSet):
        nx = 224
        ny = 224
        nz = 5
        nt = 151
        n0 = '5E'
        Folder = 'DRO_data'
    
    elif Counter('Rigid_Motion')==Counter(DataSet):
        nx = 224
        ny = 224
        nz = 5
        nt = 151
        n0 = '5E'
        Folder = 'DRO_data'
    
    elif Counter('Motion_Free')==Counter(DataSet):
        nx = 224
        ny = 224
        nz = 5
        nt = 151
        n0 = '5E'
        Folder = 'DRO_data'
    
    elif Counter('Window')==Counter(DataSet):
        nx = 251
        ny = 181
        nz = 1
        nt = 181
        n0 = '8E'
        Slice = 0
        Folder = 'Patient_2D'
    
    elif Counter('RIFF_1')==Counter(DataSet):
        nx = 480
        ny = 480
        nz = 3
        nt = 159
        n0 = '15E'
        Slice = 1
        Folder = 'Patient_2D'
    
    elif Counter('RIFF_C02')==Counter(DataSet):
        nx = 480
        ny = 480
        nz = 3
        nt = 249
        n0 = '15E'
        Slice = 1
        Folder = 'Patient_2D'

    elif Counter('RIFF_C03')==Counter(DataSet):
        nx = 480
        ny = 480
        nz = 3
        nt = 249
        n0 = '19E'
        Slice = 1
        Folder = 'Patient_2D'

    elif Counter('RIFF_C04')==Counter(DataSet):
        nx = 512
        ny = 512
        nz = 3
        nt = 249
        n0 = '17E'
        Slice = 1
        Folder = 'Patient_2D'

    elif Counter('RIFF_C05')==Counter(DataSet):
        nx = 528
        ny = 528
        nz = 3
        nt = 249
        n0 = '15E'
        Slice = 1
        Folder = 'Patient_2D'

    elif Counter('RIFF_F01')==Counter(DataSet):
        nx = 480
        ny = 480
        nz = 3
        nt = 250
        n0 = '20E'
        Slice = 1
        Folder = 'Patient_2D'
    
    elif Counter('AC51a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = '10E'
        Folder = 'Patient_3D'

    elif Counter('BA50a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = '15E'
        Folder = 'Patient_3D'

    elif Counter('CH6a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = '11E'
        Folder = 'Patient_3D'

    elif Counter('CW8a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = '15E'
        Folder = 'Patient_3D'

    elif Counter('EA14a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = '4E'
        Folder = 'Patient_3D'

    elif Counter('FJ14a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = '12E'
        Folder = 'Patient_3D'

    elif Counter('GD7a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = '9E'
        Folder = 'Patient_3D'
    
    elif Counter('HD3a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = '9E'
        Folder = 'Patient_3D'
    else:
        print("Unknown Data Set")
        print("Exiting...")
        return

    Dir = os.path.join(MDR_path('Data'),Folder,DataSet)

    img = Dir + '_DCE.dat'
    aif =  Dir + '_AIF.txt'


    data = np.fromfile(img)
    
    data = np.reshape(data,(nx,ny,nz,nt))

    print(data.shape)
    return 
    # open(img)
    # Read Data = Data
    #TODO: Implement the rest of the code when you've played with loading .dat files
