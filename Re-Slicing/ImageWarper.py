import SimpleITK as sitk
import numpy as np

def WarpImage(img,dvf):
    img_array = sitk.GetArrayFromImage(img)
    dvf_array = sitk.GetArrayFromImage(dvf)

    print("Image Shape is: " + str(img_array.shape))
    print("Dvf Shape is: " + str(dvf_array.shape))

    warper = sitk.WarpImageFilter()
    warper.SetOutputParameteresFromImage(img)
    out = warper.Execute(img,dvf)

    return out


def main():
    dvf = sitk.ReadImage('/Volumes/T7/EXP1/test/DCE_NSl/OutputDir/test/output/test/P003S2/dvf.mha')
    img = sitk.ReadImage('/Volumes/T7/EXP1/test/DCE_NSl/OutputDir/test/output/test/P003S2/wimage.mha')

    WarpedImage = WarpImage(img,dvf)
    return
main()