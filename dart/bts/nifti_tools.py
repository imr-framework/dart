import matplotlib.pyplot as plt
import nibabel as nb


def load_nifti(path):
    nifti = nb.load(path)
    print(nifti.get_fdata().shape)
    return nifti


def view_nifti(nifti_slice):
    plt.imshow(nifti_slice)
    plt.show()


path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/ORIG_atlas.nii'
nifti = load_nifti(path)
nifti_image = nifti.get_fdata()
# nifti_image = nifti_image[:, :, 16: 272]
# np.save('/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/atlas_resampled.npy', nifti_image)
# view_nifti(nifti_image[:, :, 77])
