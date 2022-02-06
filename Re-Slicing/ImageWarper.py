import SimpleITK as sitk
import numpy as np
import matplotlib as plt

def ResampleImage(img,dvf):
    img_array = sitk.GetArrayFromImage(img)
    dvf_array = sitk.GetArrayFromImage(dvf)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(dvf_array)  # Or any target geometry
    resampler.SetTransform(sitk.DisplacementFieldTransform(
        sitk.Cast(dvf_array, sitk.sitkVectorFloat64)))

    warped = resampler.Execute(img_array)
    return warped



def WarpImage(img,dvf):
    img_array = sitk.GetArrayFromImage(img)
    dvf_array = sitk.GetArrayFromImage(dvf)

    print("Image Shape is: " + str(img_array.shape))
    print("Dvf Shape is: " + str(dvf_array.shape))

    img = sitk.Cast(img, sitk.sitkFloat64)
    #dvf = sitk.Cast(dvf, sitk.sitkFloat32)

    warper = sitk.WarpImageFilter()
    warper.SetOutputParameteresFromImage(img)
    out = warper.Execute(dvf,img)

    return out


def main():
    dvf = sitk.ReadImage('/Volumes/T7/EXP3/EXP3/output/test/P003S2/dvf.mha')
    img = sitk.ReadImage('/Volumes/T7/EXP3/Test/images/P003S2.mha')

    #ResampledImage = ResampleImage(img,dvf)
    WarpedImage = WarpImage(img,dvf)
    return
main()