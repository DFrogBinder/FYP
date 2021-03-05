import os
import tqdm
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import pydicom
import platform
from tqdm import tqdm 

def chunks(lst, n):
                """
                Yield successive n-sized chunks from lst.   
                https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
                """
                for i in range(0, len(lst), n):
                        yield lst[i:i + n]

def add(number):
        return number+1

def AquisitionTimeSort(Data):
        for i in range(len(Data)):
                for j in Data[str(i)]:
                        if j.AcquisitionTime not in Data:
                                print('Hello')

def InstanceNumberSort(Data):
        Test = []
        CurrentInstance = 1
        flag = True
        while flag:
                for i,j in zip(Data,range(len(Data))):
                        if i.InstanceNumber == add(CurrentInstance):
                                CurrentInstance = j
                                Test.append(i.pixel_array)
                                if len(Test)==len(Data):
                                        flag = False

def Convert():
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

        # Detects operating system and sets the paths to the DICOMs
        if platform.system() == "Windows":
                PathDicom = r'D:\IDL\Data\Leeds_Patient_10\30\DICOM'
        elif platform.system() == "Darwin":
                PathDicom = "/Users/boyanivanov/Desktop/FYP/DICOM"
        elif platform.system() == "Linux":
                PathDicom = "/home/quaz/Desktop/FYP/DICOM"

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

        locationMatrix = np.asarray(locationMatrix) 
        locationMatrix = np.unique(locationMatrix)

        Aqu = np.asarray(Aqu) 
        Aqu = np.unique(Aqu)

        print("Sorting Data...")
        for index,location in zip(tqdm(range(len(locationMatrix))),locationMatrix):
                key =str(index)
                SortedImages[key]=[]
                for image in Data:
                        if float(image.SliceLocation) == location:
                                SortedImages[key].append(image.pixel_array)
      
        Indecies=list(chunks(range(0, 140), 5))
   
        for i in SortedImages:
                for j,k in zip(SortedImages[i],range(len(SortedImages[i]))):
                        if k not in SortedNifti:
                                SortedNifti[k]=[]
                                SortedNifti[k].append(j)
                        else:
                                SortedNifti[k].append(j)

        print('Exporting Data...')
        for nifti in tqdm(SortedNifti):
                File = np.asarray(SortedNifti[nifti]).T
                ni = nib.Nifti1Image(np.flipud(File),affine=np.eye(4))
                        
                if os.path.exists(os.path.join('','Nifti_Export')):
                        nib.save(ni, os.path.join('Nifti_Export', ['Slice'+str(nifti)+'.nii.gz'][0]))
                else:
                        os.mkdir(os.path.join(os.getcwd(),'Nifti_Export'))
                        nib.save(ni, os.path.join('Nifti_Export', ['Slice'+str(nifti)+'.nii.gz'][0]))

        print("Data is exported!")

Convert()