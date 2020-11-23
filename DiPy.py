import numpy as np
from dipy.io.image import load_nifti, save_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dti as dti
from dipy.data import get_fnames
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import fractional_anisotropy, color_fa

# Download Smaple Data
hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

data, affine = load_nifti(hardi_fname)

bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs)

# Mask the data
maskdata, mask = median_otsu(data, vol_idx=range(10, 50), median_radius=3,
                             numpass=1, autocrop=True, dilate=2)

print('data.shape (%d, %d, %d, %d)' % data.shape)
print('maskdata.shape (%d, %d, %d, %d)' % maskdata.shape)

# Instanstiate tensor model
tenmodel = dti.TensorModel(gtab)

# Fit the data
tenfit = tenmodel.fit(maskdata)

print('Computing anisotropy measures (FA, MD, RGB)')
FA = fractional_anisotropy(tenfit.evals)

# Remove background values
FA[np.isnan(FA)] = 0