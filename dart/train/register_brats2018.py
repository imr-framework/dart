from pathlib import Path
from time import time

import SimpleITK as sitk
import pandas as pd

"""
1) Extract IDs of subjects reported on
2) Register corresponding BraTS2018 volumes to atlas
"""

PATH_ATLAS = Path(r'C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\4MICCAI_2020\atlas.nii')
PATH_BRATS_DATA = Path(
    r'C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\MICCAI_BraTS_2018_Data_Training')
PATH_REGISTERED_DATA = Path(
    r'C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\4MICCAI_2020\DART_Nifti_reg')
PATH_REPORT = Path(r'../train/REPORT_ZENAS.xlsx')  # <-- CHANGE ME


def register(moving: str, fixed: str):
    moving_image = sitk.ReadImage(moving, sitk.sitkFloat32)
    fixed_image = sitk.ReadImage(fixed, sitk.sitkFloat32)

    # Initial alignment
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Similarity3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))

    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())
    return moving_resampled


# =========
# EXTRACT IDs OF SUBJECTS REPORTED ON
# =========
df = pd.read_excel(PATH_REPORT)
arr_reported_subject_idx = df['Subject ID'].astype('str')[-3:]
arr_files = list(Path(PATH_BRATS_DATA).glob('*/*'))  # List of all files in the BraTS 2018 dataset
arr_subject_idx = list(map(lambda filename: str(filename).split('_')[-2], arr_files))
dict_subject_idx_paths = dict(zip(arr_subject_idx, arr_files))
arr_reported_paths = list(map(lambda subject_id: dict_subject_idx_paths[subject_id], arr_reported_subject_idx))

# =========
# REGISTER CORRESPONDING BraTS2018 VOLUMES TO ATLAS
# =========
arr_exec_times = []  # Track execution times
total = len(arr_reported_paths)  # Number of subjects
for index, path_subject in enumerate(arr_reported_paths):
    subject = path_subject.stem  # Extract subject
    print(f'Register {subject}... {index + 1}/{total}')
    tumor_type = path_subject.parent.stem  # Extract tumor type
    files = list(path_subject.glob('*.nii.gz'))  # Get all contrasts per subject
    start = time()
    for filename in files:
        if 'seg' not in str(filename):
            registered_vol = register(moving=str(filename), fixed=str(PATH_ATLAS))
            path_output = PATH_REGISTERED_DATA / tumor_type / subject
            if not path_output.exists():  # Make save directories if they don't exist
                path_output.mkdir(parents=True)
            path_output = path_output / (filename.stem + '.gz')
            stop = time()
            sitk.WriteImage(registered_vol, str(path_output))
    arr_exec_times.append(stop - start)

total_exec_time = sum(arr_exec_times)
mean_exec_time = total_exec_time / total
print(f'Total execution time: {total_exec_time}s, mean execution time: {mean_exec_time}s')
