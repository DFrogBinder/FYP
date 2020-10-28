import numpy as np
import pandas as pd
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

def PMI_ReadPlot(File):
    s = open(File, 'r')
    TempList = []
    DataList = s.readlines()
    for entry in DataList:
        TempList.append(entry.rstrip())
    s.close()

    PDList = pd.DataFrame(TempList)
    ArrList = PDList.to_numpy()

    SplitPoint = int(np.where(ArrList=="Y-values")[0])

    tX,tY=np.split(ArrList,[SplitPoint])
    DelIndex = [0,0]
    for index in DelIndex:
        tX = np.delete(tX,index)
        tY = np.delete(tY,index)

    tX = np.delete(tX,len(tX)-1) #TODO: figure a better way of doing this

    if len(tY)==len(tX):
        for value in range(len(tX)):
            tX[value] = np.float32(tX[value])
            tY[value] = np.float32(tY[value])
    else:
        print("X and Y dimentions are not equal")
        return
        
    return tX, tY

def Load_Data(DataSet):
    if Counter('NonRigid_Motion')==Counter(DataSet):
        nx = 224
        ny = 224
        nz = 5
        nt = 151
        n0 = 5
        Folder = 'DRO_data'
    
    elif Counter('Rigid_Motion')==Counter(DataSet):
        nx = 224
        ny = 224
        nz = 5
        nt = 151
        n0 = 5
        Folder = 'DRO_data'
    
    elif Counter('Motion_Free')==Counter(DataSet):
        nx = 224
        ny = 224
        nz = 5
        nt = 151
        n0 = 5
        Folder = 'DRO_data'
    
    elif Counter('Window')==Counter(DataSet):
        nx = 251
        ny = 181
        nz = 1
        nt = 181
        n0 = 8
        Slice = 0
        Folder = 'Patient_2D'
    
    elif Counter('RIFF_1')==Counter(DataSet):
        nx = 480
        ny = 480
        nz = 3
        nt = 159
        n0 = 15
        Slice = 1
        Folder = 'Patient_2D'
    
    elif Counter('RIFF_C02')==Counter(DataSet):
        nx = 480
        ny = 480
        nz = 3
        nt = 249
        n0 = 15
        Slice = 1
        Folder = 'Patient_2D'

    elif Counter('RIFF_C03')==Counter(DataSet):
        nx = 480
        ny = 480
        nz = 3
        nt = 249
        n0 = 19
        Slice = 1
        Folder = 'Patient_2D'

    elif Counter('RIFF_C04')==Counter(DataSet):
        nx = 512
        ny = 512
        nz = 3
        nt = 249
        n0 = 17
        Slice = 1
        Folder = 'Patient_2D'

    elif Counter('RIFF_C05')==Counter(DataSet):
        nx = 528
        ny = 528
        nz = 3
        nt = 249
        n0 = 15
        Slice = 1
        Folder = 'Patient_2D'

    elif Counter('RIFF_F01')==Counter(DataSet):
        nx = 480
        ny = 480
        nz = 3
        nt = 250
        n0 = 20
        Slice = 1
        Folder = 'Patient_2D'
    
    elif Counter('AC51a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = 10
        Folder = 'Patient_3D'

    elif Counter('BA50a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = 15
        Folder = 'Patient_3D'

    elif Counter('CH6a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = 11
        Folder = 'Patient_3D'

    elif Counter('CW8a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = 15
        Folder = 'Patient_3D'

    elif Counter('EA14a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = 4
        Folder = 'Patient_3D'

    elif Counter('FJ14a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = 12
        Folder = 'Patient_3D'

    elif Counter('GD7a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = 9
        Folder = 'Patient_3D'
    
    elif Counter('HD3a')==Counter(DataSet):
        nx = 128
        ny = 128
        nz = 20
        nt = 125
        n0 = 9
        Folder = 'Patient_3D'
    else:
        print("Unknown Data Set")
        print("Exiting...")
        return

    Dir = os.path.join(MDR_path('Data'),Folder,DataSet)

    img = Dir + '_DCE.dat'
    aif =  Dir + '_AIF.txt'


    data = np.fromfile(img, dtype=np.float32)
    
    data = np.reshape(data,(nx,ny,nz,nt))

    if Counter('Patient_2D')==Counter(Folder):
        data = data[:,:,Slice,:]
        data = np.transpose(data,[2,0,1])
    elif Counter('Patient_3D')==Counter(Folder):
        data = np.transpose(data,[3,0,1,2])
    elif Counter('DRO_data')==Counter(Folder):
        data = np.transpose(data,[3,0,1,2])
    else:
        print("Folder not recognised")
        return
    time,Sa = PMI_ReadPlot(aif)   
    
    Caif = Sa - np.sum(Sa[range(n0)],dtype=np.float32)/n0
    
    return time, Caif, data, n0

def Export_Gif(Image, File, Range):
    #TODO: Find what the "PERC" routine is
    file_dat = File + '.dat'
    ImageC = Image.copy(order='C')
    
    with open(file_dat,'wb') as fout:
        fout.write(np.float32(ImageC))
        fout.close()
    
    NumberOfDimentions = len(Image.shape)
    return
