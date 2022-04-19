from ast import parse
import os
import argparse
from convert import Convert
import zipfile

def SliceData(PathToZip,Mode):
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

        Convert(PathToDicom,Mode)
        
parser = argparse.ArgumentParser(description='Process data path.')
parser.add_argument('-p',type=str,help="""Provide path to the folder containing the .zip 
                                        files of the patients""")
parser.add_argument('-m',help="Modes can be Train or Test (Case-sensitive)")                                       

if __name__ == '__main__':
    args = parser.parse_args()
    # Uncoment to provide hard-coded path and for debugging 
    # args.p = '/Users/boyanivanov/Documents/Temp_Data/ML_Data/'
    SliceData(args.p,args.m)