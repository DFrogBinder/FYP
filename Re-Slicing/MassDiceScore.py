import numpy
import pandas as pd
import nibabel as nib
from Dice_Coef import main

def Load_data(PathToFile):
    SegData = nib.load(PathToFile).get_data()
    return SegData

def main(PathToPatient):
    Data = Load_data(PathToPatient)
    ListOfMetrics = []
