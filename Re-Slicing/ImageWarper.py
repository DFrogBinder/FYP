import SimpleITK as sitk



def WarpImage(img,dvf):
    img = sitk.ReadImage("cthead1-Float.mha")
    dis = sitk.ReadImage("cthead1-dis1.nrrd")
    warper = sitk.WarpImageFilter()
    warper.SetOutputParameteresFromImage(img)
    out = warper.Execute(img,dis)

    return out


def main():
    dvf = sitk.ReadImage('')
    img = sitk.ReadImage('')

    WarpedImage = WarpedImage(img,dvf)
    return