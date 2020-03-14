import nibabel as nb

from dart.utils.maskplot3d import maskplot3d

subject = 'Brats18_TCIA10_103_1'
path = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\4MICCAI_2020\DART_Nifti_reg\LGG\{}\{}_t1.nii.gz".format(
    subject, subject)
mask_path = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\4MICCAI_2020\DART_Nifti_seg\LGG\{}.nii.gz".format(
    subject)

vol = nb.load(path).get_fdata()
mask = nb.load(mask_path).get_fdata()
maskplot3d(vol, mask)
