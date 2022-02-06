#!/usr/bin/env python

import itk
import argparse

parser = argparse.ArgumentParser(description="Wrap An Image Using A Deformation Field.")
parser.add_argument("input_image")
parser.add_argument("displacement_field")
parser.add_argument("output_image")
args = parser.parse_args()

Dimension = 2

VectorComponentType = itk.F
VectorPixelType = itk.Vector[VectorComponentType, Dimension]

DisplacementFieldType = itk.Image[VectorPixelType, Dimension]

PixelType = itk.UC
ImageType = itk.Image[PixelType, Dimension]

reader = itk.ImageFileReader[ImageType].New()
reader.SetFileName(args.input_image)

fieldReader = itk.ImageFileReader[DisplacementFieldType].New()
fieldReader.SetFileName(args.displacement_field)
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
writer.SetFileName(args.output_image)

writer.Update()
