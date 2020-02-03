import nibabel as nb
import numpy as np

from dart.utils.plot3d import plot3d

path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/MICCAI_BraTS_2018_Data_Training/HGG/Brats18_2013_5_1/Brats18_2013_5_1_t2.nii.gz'
path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/DART_Nifti_seg/HGG/Brats18_2013_5_1.nii.gz'
v = nb.load(path)
n = v.get_fdata()
plot3d(n)
