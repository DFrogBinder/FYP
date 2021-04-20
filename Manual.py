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
def MhdToNifti(PathToFolder,PathToFile,counter):
    File = load_itk(PathToFile)
    ni = nib.Nifti1Image(File.T,affine=np.eye(4))
    nib.save(ni,os.path.join(PathToFolder,['Resut'+str(counter)+'.nii.gz'][0]))
    return ni

def SaveImages(Images):
    print('Exporting Data...')
    counter = 0
    for nifti in tqdm(Images):
            File = np.asarray(nifti).T
            ni = nib.Nifti1Image(File,affine=np.eye(4))
                        
            if os.path.exists(os.path.join('','Nifti_Export')):
                    nib.save(ni, os.path.join('Nifti_Export', ["Slice"+str(counter)+'.nii.gz'][0]))
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
    cmd = [ 'elastix', '-f', fixed_image_path,'-m', moving_image_path, '-out', output_dir, '-p', parameters_file]
    try:
        subprocess.check_call(cmd)
    except:
        print ('Image registration failed')
        print (sys.exc_info())
        return

def Transformix_Call(output_dir, transform_parameters_file):
    cmd = [ 'transformix', '-def', 'all', '-out', output_dir, '-tp', transform_parameters_file]
    try:
        subprocess.check_call(cmd)
    except:
        print ('Transformix failed')
        print (sys.exc_info())
def RenameResultFile(Contents,Counter):
    os.rename('Nifti_Export\\result.0.mhd','Nifti_Export\\Result'+str(Counter)+'.mhd')

def main(path):
    ImageMatrix=[]
    # Cleans up export folder and slices the data
    Convert('D:\\IDL\\Data\\Leeds_Patient_10\\19\\DICOM')

    #Looks at the contents of the folder after data is sliced
    Folder_Contents = os.listdir(path)
    MHDImages = GetImagesMHD(Folder_Contents)
    OG_GZ=[]

    # Reduce Number of Dimetions from [384,384,1] to [384,384]
    for gz in Folder_Contents:
        im = load_itk(os.path.join(path,gz))
        #im = np.reshape(im,[im.shape[1],im.shape[2]])
        OG_GZ.append(im)
    SaveImages(OG_GZ)

    FixedImagePath = os.path.join(path,'Slice0.nii.gz')
    Param = 'D:\\IDL\\FYP\\Param.txt'
    for nifti in range(len(Folder_Contents)-1):
        Images = MakeFilename(path, [['Slice'+str(nifti+1)+'.nii.gz'][0]])
        Elastix_Call(Images[0],FixedImagePath,path,Param)
        RenameResultFile(path,nifti)
        MhdToNifti(path,os.path.join(path,['Result'+str(nifti)+'.mhd'][0]),nifti+1)
    
    Images = MakeFilename(path,GetImagesMHD(os.listdir(path)))
    for entry in Images:
        ImageMatrix.append(load_itk(entry))
    SaveImages(ImageMatrix)
    
main('D:\\IDL\\FYP\\Nifti_Export')