import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
import sys
import subprocess
from convert import Convert
from tqdm import tqdm
import zipfile

def SliceData(PathToZip):
    PatientList = os.listdir(PathToZip)
    if '.DS_Store' in PatientList:
        os.remove(os.path.join(PathToZip,'.DS_Store'))
        
    for patient in PatientList:
        PatientFolder = os.path.join(PathToZip,patient.split('.')[0])
        os.mkdir(PatientFolder)

        with zipfile.ZipFile(os.path.join(PathToZip,patient), 'r') as zip_ref:
            zip_ref.extractall(PatientFolder)

        PathToScans = os.path.join(PatientFolder,'scans')
        tFolder = os.listdir(PathToScans)
        PathToDicom = os.path.join(os.path.join(os.path.join(PathToScans,tFolder[0]),os.listdir(os.path.join(PathToScans,tFolder[0]))[0]))

        Convert(PathToDicom,'Train')
SliceData('/Users/boyanivanov/Documents/Temp_Data/ML_Data')