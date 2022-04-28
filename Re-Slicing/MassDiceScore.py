import numpy
import pandas as pd
import nibabel as nib
from Dice_Coef import get_stats

def Load_data(PathToFile):
    SegData = nib.load(PathToFile).get_data()
    return SegData

def main(PathToPatient):
    Data = Load_data(PathToPatient)
    ListOfMetrics = []
    Fixed_Image = Data[:,:,1] # Set second dynamic as the fixed image 
    for i in range(Data.shape[2]):
        ListOfMetrics.append(get_stats(Fixed_Image,Data[:,:,i]))
    df = pd.DataFrame(ListOfMetrics)
    writer = pd.ExcelWriter('Stats.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='welcome', index=False)
    writer.save()
    return ListOfMetrics


main('/Volumes/T7/Internal_Report_Data/Internal_Report_Graphics/P29_Original_Segmentation/P29_Full-Segmentation.nii.gz')