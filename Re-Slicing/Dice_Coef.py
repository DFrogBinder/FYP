import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.spatial
import sys
import argparse
from PIL import Image


def Load_data(PathToFile):
    SegData = nib.load(PathToFile).get_data()
    return SegData

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
    Data = Load_data(Path)
    Im1 = Data[:,:,32] #56 ; 32
    Im2 = Data[:,:,33] #57 ; 33

    Dice_coef = get_dice_coefficient(Im1,Im2)
    Hausdorff = get_hausdorff(Im1,Im2)
    print('Dice Coef is: ' + str(Dice_coef))
    #print("Hausdorff distance is: " + Hausdorff)
    return

parser = argparse.ArgumentParser(description='Process data path.')
parser.add_argument('-p',help='The arguments file in nifti format')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.p)