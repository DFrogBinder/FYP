import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
import sys
import subprocess
from convert import Convert
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
def GetImagesMHD(FileSet):
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
    counter = 0
    for nifti in tqdm(Images):
            File = np.asarray(nifti).T
            ni = nib.Nifti1Image(File,affine=np.eye(4))
                        
            if os.path.exists(os.path.join('','Nifti_Export')):
                    nib.save(ni, os.path.join('Nifti_Export', ["MhD_Image"+str(counter)+'.nii.gz'][0]))
                    counter = counter+1
            else:
                    os.mkdir(os.path.join(os.getcwd(),'Nifti_Export'))
                    nib.save(ni, os.path.join('Nifti_Export', ['Slice'+str(nifti)+'.nii.gz'][0]))
                    counter = counter+1
def CheckListForString(List,String):
    Flag = False
    for i in range(len(List)):
        if String in List[i]:
            Flag = True
        else:
            pass
    return Flag
def Elastix_Call(moving_image_path,fixed_image_path, output_dir, parameters_file):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    cmd = [ 'elastix', '-m', moving_image_path, '-f', fixed_image_path, '-out', output_dir, '-p', parameters_file]
    try:
        subprocess.check_call(cmd)
    except:
        print ('Image registration failed')
        print (sys.exc_info())

def Transformix_Call(output_dir, transform_parameters_file):
    cmd = [ 'transformix', '-def', 'all', '-out', output_dir, '-tp', transform_parameters_file]
    try:
        subprocess.check_call(cmd)
    except:
        print ('Transformix failed')
        print (sys.exc_info())
def RenameResultFile(Contents,Counter):
    os.rename('Nifti_Export\\result.0.mhd','Nifti_Export\\result'+str(Counter)+'.mhd')

def main(path):
    ImageMatrix=[]
    # Cleans up export folder and slices the data
    Convert()

    #Looks at the contents of the folder after data is sliced
    Folder_Contents = os.listdir(path)
    MHDImages = GetImagesMHD(Folder_Contents)
    PathToFixedImage=[]

    for nifti in range(len(Folder_Contents)):
        UpdatedContent = os.listdir(path)
        Param = 'D:\\IDL\\FYP\\Param.txt'
        if CheckListForString(UpdatedContent,['result'+str(nifti-1)+'.mhd'][0]):
            Images = MakeFilename(path,[['result'+str(nifti-1)+'.mhd'][0],['Slice'+str(nifti+1)+'.nii.gz'][0]])
            Elastix_Call(Images[0],PathToFixedImage,path,Param)
            RenameResultFile(path,nifti)
        else:
            Images = MakeFilename(path, [['Slice'+str(nifti)+'.nii.gz'][0],['Slice'+str(nifti+1)+'.nii.gz'][0]])
            Elastix_Call(Images[1],Images[0],path,Param)
            RenameResultFile(path,nifti)
            PathToFixedImage=Images[0]

    Images = MakeFilename(path,GetImagesMHD(os.listdir(path)))
    for entry in Images:
        ImageMatrix.append(load_itk(entry))
    SaveImages(ImageMatrix)
    
main('D:\\IDL\\FYP\\Nifti_Export')