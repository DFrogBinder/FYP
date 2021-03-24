import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import pydicom
import platform
from tqdm import tqdm

def add(number):
        return number+1

def AquisitionTimeSort(Data):
        SortedFile={}
        SortedFileRef = {}
        Test = {}
        # Creates Ref Arrays
        for depth in Data:
                key=str(depth)
                SortedFileRef[key]=[]
                for location in Data[str(depth)]:
                        SortedFileRef[key].append(float(location.AcquisitionTime))
        # Sorts Ref Arrays
        for depth in SortedFileRef:
                for array in SortedFileRef[depth]:
                        SortedFileRef[depth]=np.sort(SortedFileRef[depth])
        
        for depth in SortedFileRef:
                key=str(depth)
                SortedFile[key]=[]
                Test[key]=[]
                for array in SortedFileRef[depth]:
                        for DataSet in Data[depth]:
                                if float(DataSet.AcquisitionTime) == array:
                                        SortedFile[key].append(DataSet)
                                        Test[key].append(float(DataSet.AcquisitionTime))
        
        # Runs a simple test to highlight potential errors         
        for depth in SortedFile:
                for DataSet,index in zip(SortedFile[depth],range(len(SortedFile[depth]))):
                        if float(DataSet.AcquisitionTime)==Test[depth][index]:
                                continue
                        else:
                                print("Aquisition Times are not sorted correctly!")
                                print("Exiting...")
                                return 
                                       
        return SortedFile
def ManualRegSort(Data):

        Data = AquisitionTimeSort(Data)
        
        SingleDynamicMatrix={}
        # Select the first Slice
        SData = Data['0']

        for i in range(len(Data['0'])):
                key = str(i)
                SingleDynamicMatrix[key]=[]
        for dynamic in range(len(Data['0'])):
                SingleDynamicMatrix[str(dynamic)].append(SData[int(dynamic)].pixel_array)

        return SingleDynamicMatrix

def OneDynamicSort(Data):
       
        Data = AquisitionTimeSort(Data)

        OneDynamicSortMatrix={}

        for i in range(len(Data['0'])):
                key = str(i)
                OneDynamicSortMatrix[key]=[]
        for dynamic in range(len(Data['0'])):
                for LocationNumber in Data:
                        OneDynamicSortMatrix[str(dynamic)].append(Data[LocationNumber][int(dynamic)].pixel_array)

        return OneDynamicSortMatrix

def DGD_Sort(Data):
        DGDs ={}

        for i in range(146):
                key = str(i)
                DGDs[key]=[]
        for dynamic in range(146):
                for LocationNumber in Data:
                        DGDs[str(dynamic)].append(Data[LocationNumber][int(dynamic)].pixel_array)
        return DGDs
def CleanFolder(path):
        Contents = os.listdir(path)
        for entry in Contents:
                os.remove(os.path.join(path,entry))
def Convert(PathDicom):
        # Variable Declarations
        mhd_entry_list = []
        nifti_matrix = []
        locationMatrix = []
        InstaceMatrix=[]
        Data=[]
        SortedImages = {}
        SortedNifti={}
        Instace ={}
        Aqu=[] 
        OrderedImages=[]    
        DGD=[]   
        Bval=[]

        # Detects operating system and sets the paths to the DICOMs
        if platform.system() == "Windows":
                Parts = PathDicom.split('\\')
        elif platform.system() == "Darwin":
                Parts = PathDicom.split('/')
        elif platform.system() == "Linux":
                Parts = PathDicom.split('/')

        
        if int(Parts[len(Parts)-2])==19:
                flag = "T1"
        elif int(Parts[len(Parts)-2])==30:
                flag = "DTI"
        elif int(Parts[len(Parts)-2])==39 or int(Parts[len(Parts)-2])== 40:
                flag = "DCE"
        else:
                print("Unknown Modality!")
                print("Exiting...")
                return
        lstFilesDCM = []  # create an empty list
        for dirName, subdirList, fileList in os.walk(PathDicom):
                for filename in fileList:
                        if ".dcm" in filename.lower():  # check whether the file's DICOM
                                lstFilesDCM.append(os.path.join(dirName,filename))
                        elif ".ima" in filename.lower():
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
                Aqu.append(ds.AcquisitionTime)
                try:
                        DGD.append(ds[0x0019,0x100e].value)
                        Bval.append(ds[0x0019,0x100c].value)
                except:
                        continue
        
        # Convert to Numpy arrays and extract unique values
        locationMatrix = np.unique(np.asarray(locationMatrix))
        Aqu = np.unique(np.asarray(Aqu)) 
        DGD = np.unique(np.asarray(DGD))
        Bval = np.unique(np.asarray(Bval))
 
        print("Sorting Data...")
        for index,location in zip(tqdm(range(len(locationMatrix))),locationMatrix):
                key =str(index)
                SortedImages[key]=[]
                for image in Data:
                        if float(image.SliceLocation) == location:
                                SortedImages[key].append(image)
      
        if flag == 'T1' or flag == 'DCE':
                if flag == 'DCE':
                        SortedImages.pop(str(len(SortedImages)-1))
                SortedNifti = ManualRegSort(SortedImages)
        elif flag == 'DTI':
                SortedNifti = DGD_Sort(SortedImages)
        else:
                print("Unknown Flag value!")
                print("Exiting...")
                return

        CleanFolder(os.path.join('','Nifti_Export'))
        
        print('Exporting Data...')
        for nifti in tqdm(SortedNifti):
                File = np.asarray(SortedNifti[nifti]).T
                ni = nib.Nifti1Image(File,affine=np.eye(4))
                        
                if os.path.exists(os.path.join('','Nifti_Export')):
                        nib.save(ni, os.path.join('Nifti_Export', ['Slice'+str(nifti)+'.nii.gz'][0]))
                else:
                        os.mkdir(os.path.join(os.getcwd(),'Nifti_Export'))
                        nib.save(ni, os.path.join('Nifti_Export', ['Slice'+str(nifti)+'.nii.gz'][0]))

        print("Data is exported!")
