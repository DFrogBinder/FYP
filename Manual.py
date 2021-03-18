import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
from tqdm import tqdm

'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan

# Filteres a filset and returns only the .mhd files
def Mhd_Filter(FileSet):
    MhdImages = []
    for entry in FileSet:
        if '.mhd' in entry:
            MhdImages.append(entry)
    return MhdImages

# Creates a full pathname of the file
def MakeFilename(path,Images):
    FullPath=[]
    for image in Images:
        FullPath.append(os.path.join(path,image))
    return FullPath

def SaveImages(Images):
    print('Exporting Data...')
    for nifti in tqdm(Images):
            File = np.asarray(nifti).T
            ni = nib.Nifti1Image(File,affine=np.eye(4))
                        
            if os.path.exists(os.path.join('','Nifti_Export')):
                    nib.save(ni, os.path.join('Nifti_Export', ['Slice'+str(nifti)+'.nii.gz'][0]))
            else:
                    os.mkdir(os.path.join(os.getcwd(),'Nifti_Export'))
                    nib.save(ni, os.path.join('Nifti_Export', ['Slice'+str(nifti)+'.nii.gz'][0]))


def main(path):
    ImageMatrix=[]

    Folder_Contents = os.listdir(path)
    Images = Mhd_Filter(Folder_Contents)
    Images = MakeFilename(path,Images)
    
    for entry in Images:
        ImageMatrix.append(load_itk(entry))
    SaveImages(ImageMatrix)
    
main('D:\\IDL\\FYP\\Nifti_Export')