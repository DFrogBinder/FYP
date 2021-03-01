import os
import tqdm
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import pydicom
import platform
from tqdm import tqdm 

def dcm2nifti(path):

    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(path)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    scan = sitk.GetArrayFromImage(itkimage)

    ct_scan_nifti = nib.Nifti1Image(scan, affine=np.eye(4))

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return scan

mhd_entry_list = []
nifti_matrix = []
locationMatrix = []
InstaceMatrix=[]
Data=[]
SortedImages = {}
Instace ={}

# Detects operating system and sets the paths to the DICOMs
if platform.system() == "Windows":
        PathDicom = r'D:\IDL\FYP\DICOM\Leeds_Patient_4128010\19\DICOM'
elif platform.system() == "Darwin":
        PathDicom = "/Users/boyanivanov/Desktop/FYP/DICOM"
elif platform.system() == "Linux":
        PathDicom = "/home/quaz/Desktop/FYP/DICOM"
        
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                       lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
RefDs = pydicom.read_file(lstFilesDCM[0])

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

# The array is sized based on 'ConstPixelDims'
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

        #loop through all the DICOM files
print("Loading Data...")
for filenameDCM in tqdm(lstFilesDCM):
        # read the file
        ds = pydicom.read_file(filenameDCM)
        Data.append(ds)
        locationMatrix.append(ds.SliceLocation)

locationMatrix = np.asarray(locationMatrix) 
locationMatrix = np.unique(locationMatrix)

print("Sorting Data...")
for index,location in zip(tqdm(range(len(locationMatrix))),locationMatrix):
        key =str(index)
        SortedImages[key]=[]
        for image in Data:
                if float(image.SliceLocation) == location:
                        SortedImages[key].append(image.pixel_array)
CurrentInstance=0
def add(number):
        return number+1
Test =[]
flag = True
while flag:
        for i,j in zip(Data,range(len(Data))):
                if i.InstanceNumber==add(CurrentInstance):
                        CurrentInstance = j
                        Test.append(i.pixel_array)
                        if len(Test)==len(Data):
                                flag=False
                else:
                        continue

def chunks(lst, n):
    """Yield successive n-sized chunks from lst.   https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
Indecies=list(chunks(range(0, 140), 5))
     
print('Exporting Data...')
for i,j in zip(tqdm(Indecies),range(len(Indecies))):
        NiftiFile=[]
        for dynamic in i:
                #ni = nib.Nifti1Image(np.flipud(np.rot90(np.stack(SortedImages[image],axis=2))),affine=np.eye(4))
                NiftiFile.append(Data[dynamic].pixel_array)
        NiftiFile = np.asarray(NiftiFile)
        ni = nib.Nifti1Image(NiftiFile.T,affine=np.eye(4))
                
        if os.path.exists(os.path.join('','Nifti_Export')):
                nib.save(ni, os.path.join('Nifti_Export', ['Slice'+str(j)+'.nii.gz'][0]))
        else:
                os.mkdir(os.path.join(os.getcwd(),'Nifti_Export'))
                nib.save(ni, os.path.join('Nifti_Export', ['Slice'+str(j)+'.nii.gz'][0]))

print("Data is exported!")
