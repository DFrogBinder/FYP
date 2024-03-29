import numpy as np
import os
import matplotlib.pyplot as plt 
import pydicom
import pandas as pd 
import numpy.ma as ma
import platform 
import SimpleITK as sitk
import cv2

from sklearn.linear_model import LinearRegression
from PIL import Image as im
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pathlib import Path
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

#TODO: Update Broad.yml file to include simpleITK and openCV2


def GIF(Images):
        # Code to create .gif file of the fitted images
        print("Creating GIF image...")
        fig, ax = plt.subplots(figsize=(5, 8))
        def update(i):
                im_normed = Images[i,:,:]
                ax.imshow(im_normed,cmap='gray')
                ax.set_axis_off()
                print("Exporting Frame: "+str(i))
        anim = FuncAnimation(fig, update, frames=np.arange(0, 29), interval=1).save("Anim.gif")
        plt.close()

def LinFit(DiffImage, bv, g):
                sz = DiffImage.shape  
                
                bvalues = bv
                S = DiffImage
                # Flattens the image into a vector
                S = np.reshape(DiffImage,[sz[0]*sz[1], sz[2]],order='F')
                S[S == 0] = 1e-16  # Removes zeroes to prevent erros
                
                S[S==0]=np.amin(S[S>0])
                
                logS = np.log(S)
                logS = logS.T

                bv = np.reshape(bv,[sz[2],1])
                logS0_ADC = np.linalg.lstsq(bv,logS,rcond=None)[0].T

                S0 = np.linalg.lstsq(bv,logS,rcond=None)[1]

                logS0_ADC = np.asarray(logS0_ADC).T
                S0 = np.asarray(S0).T  
                where_are_NaNs = np.isnan(logS0_ADC)
                logS0_ADC[where_are_NaNs] = 0
                
                return logS0_ADC,S0,S
def LinPlot(X,Y):
        lr = LinearRegression()
        lr.fit(X.reshape(-1,1),Y.T)        
        Y_Pred = lr.predict(X.reshape(-1,1))
        
        ImagePixel = np.reshape(Y,[172,172,30])[80,80,:]
        Y_Pred = np.reshape(Y_Pred,[30,172,172])
        
        plt.scatter(X,ImagePixel)
        plt.plot(X,Y_Pred[:,80,80],color='red')
        plt.title("Linear Fit of the data for a single pixel")
        plt.ylabel("Signal Intesity")
        plt.xlabel('Bvalues (s/mm^2)')
        plt.show()
def Register(ImagePopulation):
        # Concatenate the ND images into one (N+1)D image
        population = ImagePopulation
        vectorOfImages = sitk.VectorOfImage()

        for filename in population:
                vectorOfImages.push_back(sitk.ReadImage(filename))

        image = sitk.JoinSeries(vectorOfImages)

        # Register
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(image)
        elastixImageFilter.SetMovingImage(image)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap('groupwise'))
        elastixImageFilter.Execute()
        
        return image
        
def Scratch():
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

        # Initialize some variables 
        Bvalues=[]
        locationMatrix = []
        Data=[]
        
        SortedImages = {}
        SortedBvals = {}
        SortedDirection = {}

        TestList = []
        TestList2 = []

        #loop through all the DICOM files
        print("Loading Data...")
        for filenameDCM in tqdm(lstFilesDCM):
                # read the file
                ds = pydicom.read_file(filenameDCM)
                Data.append(ds)
                locationMatrix.append(ds.SliceLocation)
            
        locationMatrix = np.asarray(locationMatrix) 
        locationMatrix = np.unique(locationMatrix)
        
        print("Sorting Data")
        for index,location in zip(tqdm(range(len(locationMatrix)-1)),locationMatrix):
                key = "Position_"+str(index)
                SortedImages[key]=[]
                SortedBvals[key]=[]
                SortedDirection[key]=[]
                for image in Data:
                        if float(image.SliceLocation) == location:
                                SortedImages[key].append(image.pixel_array)
                                SortedBvals[key].append(int(image[0x0019,0x100c].value))
                                if key in SortedDirection.keys():
                                        pass
                                else:
                                        SortedDirection[key].append(ds[0x0019,0x100e].value)
                        
           
        Fitted_Images =[]
        # Linear fittet
        print("Performing Liner fit")
        for image in tqdm(SortedImages):
                ImageMatrix = np.transpose(np.asarray(SortedImages[image]))
                bv = np.asarray(SortedBvals[image])
                g = np.asarray(SortedDirection[image])
                logS0_ADC,S0,S= LinFit(ImageMatrix,bv,g)
                logS0_ADC = np.reshape(logS0_ADC,[172,172])
                Fitted_Images.append(logS0_ADC)
        Fitted_Images = np.asarray(Fitted_Images)
        
        # Plot the linear fit
        # LinPlot(bv,S)

        #Creates .gif file of the images
        #GIF(Fitted_Images)

        if os.path.exists(os.path.join(os.getcwd(),'Images')):
                print("Image Folder Already Exists")
                print("Saving Images to Folder...")
                for image in range(len(Fitted_Images[:,1,1])):
                        name = os.path.join(os.path.join(os.getcwd(),'Images'),str(image)+".png")
                        cv2.imwrite(name, Fitted_Images[image,:,:]*10000)
        else:
                print("Image Foleder Not Found!")
                print("Saving Images to Folder...")
                os.mkdir(os.path.join(os.getcwd(),'Images'))
                print("Saving Images...")
                for image in range(len(Fitted_Images[:,1,1])):
                        name = os.path.join(os.path.join(os.getcwd(),'Images'),str(image)+".png")
                        cv2.imwrite(name, Fitted_Images[image,:,:]*10000)
        
        ImageFiles = os.listdir(os.path.join(os.getcwd(),'Images'))
        RegImagePath=[]
        for File in ImageFiles:
                path = os.path.join(os.path.join(os.getcwd(),'Images'), File)
                RegImagePath.append(path)
        reg = Register(RegImagePath)
        print("Done")
        
        
Scratch()



 
