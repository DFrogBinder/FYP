import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import pydicom
import platform
import matplotlib.pyplot as plt
import PIL
import cv2
import argparse
from SimpleITK.SimpleITK import Normalize
from tqdm import tqdm

def normalize_image(Data,viz=False): # disp is an array in uint8 data type
        
        norm = np.zeros((Data.shape[0],Data.shape[1]))
        disp_norm = cv2.normalize(Data,norm,0,255,cv2.NORM_MINMAX)
        
        if viz:
                plt.imshow(disp_norm)
                plt.show()

        return disp_norm 

def get_sitk_image_details_from_DICOM(filenameDCM):
        """ Reads and returns image spacing of the input DICOM File.
        Args:
        ----
        filenameDCM (string): File path to a DICOM File.
        Returns:
        -------
        spacing (float): Float value describing the space between pixels in the given image.
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(filenameDCM)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        spacing = image.GetSpacing()
        return spacing

def convert_to_mha(Data):
        image = sitk.ReadImage(Data)
def Add_Noise(Data,Dim):
    noise = np.random.normal(0,1,[Dim[0],Dim[1]])
    return Data + noise
def add(number):
        return number+1
def SingleSliceSort(Data):
     Images=[]
     Data = Data['3']
     for i in Data:   
             Images.append(i.pixel_array)
     return Images

def GroupedImageSort(Data,flag):
        Images={}
        if flag == 'DCE':
                itr = len(Data['0'])
        else:
                itr = 28

        for i in range(itr):
                Images[i]=[]
                for d in Data:
                        tVar = Data[d]
                        Images[i].append(tVar[i].pixel_array)
        return Images

def DTI_Image_Sort(Data):
        Data = AquisitionTimeSort(Data)
        Images={}
        NumberOfImages=len(Data['0'])
        for i in range(len(Data)):
                Images[i]=[]
                for d in range(NumberOfImages):
                        tVar = Data[str(i)][d]
                        tPixelArray = tVar.pixel_array
                        Images[i].append(tPixelArray)
        return Images               

def FixedImageSort(Data):
        FF = {}
        SF = {}

        for i in range(5):
                FF[str(i)]=[]
                SF[str(i)]=[]
        # Sets the first file as fixed images
        for i in Data:
                for j in range(len(Data[i])-1):
                        Augmented_Fixed_Image = Add_Noise(Data[i][0].pixel_array,Data[i][0].pixel_array.shape)
                        FF[i].append(Augmented_Fixed_Image)
                        SF[i].append(Data[i][j+1].pixel_array)
        #SF.pop(0)
        return SF,FF  
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
def FlattenDickt(Data):
        Flat=[]
        for i in range(len(Data)):
                for j in range(len(Data[str(i)])):
                        Flat.append(Data[str(i)][j].pixel_array)
        return Flat
def UNetSort(Data):
        Data = AquisitionTimeSort(Data)
        SingleDynamicMatrix={}
        SliceMatrix={}
        FlatData = FlattenDickt(Data)
        for i in range(len(Data['0'])):
                key=str(i)
                SliceMatrix[key]=[]

        for dynamic in SliceMatrix:
                Indecies = np.arange(int(dynamic),len(FlatData),28)
                for index in Indecies:
                        SliceMatrix[dynamic].append(FlatData[index])
        return SliceMatrix

def ManualRegSort(Data):

        Data = AquisitionTimeSort(Data)
        
        SingleDynamicMatrix={}
        # Select the first Slice
        SData = Data['3']

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
def Convert(PathDicom,Mode):
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

        
        if (str(Parts[len(Parts)-2])==[str(19)+'_unknown'][0] or 
                str(Parts[len(Parts)-2])==[str(20)+'_unknown'][0] or 
                str(Parts[len(Parts)-2])==[str(22)+'_unknown'][0] or 
                str(Parts[len(Parts)-2])==[str(23)+'_unknown'][0] or 
                str(Parts[len(Parts)-2])==[str(25)+'_unknown'][0] or 
                str(Parts[len(Parts)-2])==[str(29)+'_unknown'][0]):
                flag = "T1"
        elif (str(Parts[len(Parts)-2])==[str(30)+'_unknown'][0] or 
                str(Parts[len(Parts)-2])==[str(31)+'_unknown'][0]):
                flag = "DTI"
        elif (str(Parts[len(Parts)-2])==[str(38)+'_unknown'][0] or
              str(Parts[len(Parts)-2])==[str(39)+'_unknown'][0] or 
                str(Parts[len(Parts)-2])==[str(40)+'_unknown'][0] or
                str(Parts[len(Parts)-2])==[str(41)+'_unknown'][0] or
                str(Parts[len(Parts)-2])==[str(42)+'_unknown'][0] or
                str(Parts[len(Parts)-2])==[str(43)+'_unknown'][0] or
                str(Parts[len(Parts)-2])==[str(44)+'_unknown'][0] or
                str(Parts[len(Parts)-2])==[str(45)+'_unknown'][0] or
                str(Parts[len(Parts)-2])==[str(46)+'_unknown'][0] or
                str(Parts[len(Parts)-2])==[str(47)+'_unknown'][0] or
                str(Parts[len(Parts)-2])==[str(48)+'_unknown'][0] or
                str(Parts[len(Parts)-2])==[str(49)+'_unknown'][0] or
                str(Parts[len(Parts)-2])==[str(51)+'_unknown'][0] or
                str(Parts[len(Parts)-2])==[str(52)+'_unknown'][0]):
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
        Spacing_ITK = get_sitk_image_details_from_DICOM(PathDicom)
        
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

        #SF,FF = FixedImageSort(SortedImages['3'])
        
        
        if flag == 'T1' or flag == 'DCE':
                if flag == 'DCE':
                        SortedImages.pop(str(len(SortedImages)-1))
                SortedNifti = DTI_Image_Sort(SortedImages)
        elif flag == 'DTI':
                SortedNifti = DTI_Image_Sort(SortedImages)
        else:
                print("Unknown Flag value!")
                print("Exiting...")
                return
        cwd = os.getcwd()
        #CleanFolder(os.path.join(cwd,'Nifti_Export'))
        
        print('Exporting Data...')
        for i,nifti in zip(range(0,len(SortedNifti)),tqdm(SortedNifti)):
                if Mode != 'Train': 
                        File = np.asarray(SortedNifti[str(i)]).T
                        ni = nib.Nifti1Image(File,affine=np.eye(4))   
                        ResultFolder = os.path.join(PathDicom,'Nifti_Export') 
                        if os.path.exists(os.path.join('','Nifti_Export')):
                                nib.save(ni, os.path.join('Nifti_Export', ['Slice'+str(i)+'.nii.gz'][0]))
                        else:
                                os.mkdir(os.path.join(os.getcwd(),'Nifti_Export'))
                                nib.save(ni, os.path.join('Nifti_Export', ['Slice'+str(i)+'.nii.gz'][0]))
                elif Mode == 'Train':
                        try:
                                itr = SortedNifti[str(i)]
                        except:
                                itr = SortedNifti[i]
                                        
                        for Slice in itr:
                                File = np.asarray(itr).astype(np.float32)
                                for j in range(File.shape[0]):
                                        normFile = normalize_image(File[j,:,:])
                                        File[j,:,:] = normFile
                                File = sitk.GetImageFromArray(File)
                                File.SetSpacing(Spacing_ITK)
                                writer = sitk.ImageFileWriter()
                                ResultFolder = os.path.join(PathDicom,'Nifti_Export')
                                if os.path.exists(ResultFolder):
                                        writer.SetFileName(os.path.join(ResultFolder, ['Slice_'+str(i)+'.mha'][0]))
                                        writer.Execute(File)
                                                
                                else:
                                        os.mkdir(ResultFolder)
                                        writer.SetFileName(os.path.join(ResultFolder, ['Slice_'+str(i)+'.mha'][0]))
                                        writer.Execute(File)
                                        
                else:
                        print('Unrecognised mode, exiting...')
                        return

        print("Data is exported to "+str(ResultFolder))
