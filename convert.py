import os
import tqdm
import nibabel as nib
import numpy as np
import SimpleITK as sitk


def mhd2nifti(path,name):

    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(path)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    scan = sitk.GetArrayFromImage(itkimage)

    ct_scan_nifti = nib.Nifti1Image(scan, affine=np.eye(4))

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    if os.path.exists(os.path.join(os.getcwd(),'Nifti_Export')):
        nib.save(ct_scan_nifti, os.path.join('Nifti_Export', [str(index)+'.nii.gz'][0]))
    else:
        os.mkdir(os.path.join(os.getcwd(),'Nifti_Export'))
        nib.save(ct_scan_nifti, os.path.join('Nifti_Export', [str(index)+'.nii.gz'][0]))

    #return ct_scan_nifti, origin, spacing

imagepath = r'D:\IDL\FYP\DICOM\Leeds_Patient_4128010\19\DICOM\T1_slice_3_MoCoMo\Fitted'
dir_entry = os.listdir(imagepath)
mhd_entry_list = []

for entry in dir_entry:
    if ".mhd" in entry:
        mhd_entry_list.append(entry)

for mhd_file,index in zip(mhd_entry_list,range(len(mhd_entry_list))):
    mhd2nifti(os.path.join(imagepath,mhd_file),index)


print("Saving Nifti is done")