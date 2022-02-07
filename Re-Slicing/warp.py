import tqdm
import itk
import sys
import os
import argparse

# sys.argv=['warp.py','P003S2.mha', 'dvf.mha', 'WImage.mha']

# parser = argparse.ArgumentParser(description="Wrap An Image Using A Deformation Field.")
# parser.add_argument("input_image")
# parser.add_argument("displacement_field")
# parser.add_argument("output_image")
# args = parser.parse_args()
def warp(input_image,displacement_field,output_image):
    
    Dimension = 3

    VectorComponentType = itk.F
    VectorPixelType = itk.Vector[VectorComponentType, Dimension]

    DisplacementFieldType = itk.Image[VectorPixelType, Dimension]

    PixelType = itk.UC
    ImageType = itk.Image[PixelType, Dimension]

    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(input_image)

    fieldReader = itk.ImageFileReader[DisplacementFieldType].New()
    fieldReader.SetFileName(displacement_field)
    fieldReader.Update()

    deformationField = fieldReader.GetOutput()

    warpFilter = itk.WarpImageFilter[ImageType, ImageType, DisplacementFieldType].New()

    interpolator = itk.LinearInterpolateImageFunction[ImageType, itk.D].New()

    warpFilter.SetInterpolator(interpolator)

    warpFilter.SetOutputSpacing(deformationField.GetSpacing())
    warpFilter.SetOutputOrigin(deformationField.GetOrigin())
    warpFilter.SetOutputDirection(deformationField.GetDirection())

    warpFilter.SetDisplacementField(deformationField)
    warpFilter.SetInput(reader.GetOutput())

    writer = itk.ImageFileWriter[ImageType].New()
    writer.SetInput(warpFilter.GetOutput())
    writer.SetFileName(output_image)

    writer.Update()
def main(path):
   List = os.listdir(path)
   if 'Warped_Images' in List:
       print('Output directory already exist')
   else:
       os.mkdir(os.path.join(path, 'Warped_Images'))
       
   if 'Test' in List and 'dvfs' in List:
        InputImagesDir = os.path.join(path,'Test','images')
        InputImageList = os.listdir(InputImagesDir)
        
        DisplacementFieldDir = os.path.join(path,'dvfs')
        DisplacementFieldList = os.listdir(DisplacementFieldDir)
        for image in tqdm.tqdm(InputImageList):
            Input_image = os.path.join(InputImagesDir,str(image))
            if str(image).split('.')[0] in DisplacementFieldList:
                displacement_field = os.path.join(DisplacementFieldDir,str(image).split('.')[0],'dvf.mha')
            OutputImage = os.path.join(path,'Warped_Images',str('Warped_'+image))
            warp(Input_image,displacement_field,OutputImage)

main('/Volumes/T7/EXP3/')