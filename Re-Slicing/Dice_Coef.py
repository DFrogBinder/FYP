import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.spatial
import sys
import cv2
import argparse
from PIL import Image

def Load_data(PathToFile):
    SegData = nib.load(PathToFile).get_data()
    return SegData

def Array2ListOfSets(Array):
  result = []
  for i in range(Array.shape[-1]-1):
    result.append(list(Array[:,:,i]))
  return result

def intersect(Data):
  LData = Array2ListOfSets(Data)
  result = []
  for i in range(Data.shape[-1]-1):
    result.append(cv2.bitwise_and(Data[:,:,i],Data[:,:,i+1]))
  return result

def get_extended_dice(Data):
  G = Data.shape[-1]
  IntersectOfSet = np.asanyarray(intersect(Data))
  DiceG = G*(np.abs(IntersectOfSet)/np.abs(Data.sum(axis=2,keepdims=True)))
  return Data 

def get_dice_coefficient(mask_gt, mask_pred):
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum 
def get_hausdorff(Im1,Im2):
  distance = scipy.spatial.distance.directed_hausdorff(Im1,Im2)
  return distance

def main(Path):
  # Check is we have a full segmentation 
    Data = Load_data(Path)
    if Data[:,:,1].any()==False:
      print('The file provided does not contain a full segmentation!')
    else:
      Data = get_extended_dice(Data)

    #Im1 = Data[:,:,56] #56 ; 32
    #Im2 = Data[:,:,57] #57 ; 33


    Dice_coef = get_dice_coefficient(Im1,Im2)
    Hausdorff = get_hausdorff(Im1,Im2)
    print('Dice Coef is: ' + str(Dice_coef))
    print("Hausdorff distance is: " + str(Hausdorff[0]))
    return Dice_coef, Hausdorff

parser = argparse.ArgumentParser(description='Process data path.')
parser.add_argument('-p',type=str,help='The arguments file in nifti format')
#parser.add_argument('-f',action='store_true',help="""Set flag only if the full
#                                            set of images are segmented.""")

if __name__ == '__main__':
    args = parser.parse_args()
    # Uncoment to provide hard-coded path and for debugging 
    # args.p = '/Volumes/T7/Internal_Report_Data/Internal_Report_Graphics/P29_Original_Segmentation/P29_Full-Segmentation.nii.gz'
    main(args.p)
