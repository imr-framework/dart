import os

import SimpleITK as sitk


def register(moving_dir: str, fixed: str, output_dir: str):
    moving_all_mod = [os.path.join(moving_dir, x) for x in os.listdir(moving_dir)]  # Paths to all modalities

    for i, m in enumerate(moving_all_mod):
        moving_image = sitk.ReadImage(m, sitk.sitkFloat32)
        fixed_image = sitk.ReadImage(fixed, sitk.sitkFloat32)

        # Initial alignment
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(),
                                                              sitk.CenteredTransformInitializerFilter.GEOMETRY)
        # moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, 0.0,
        #                                  moving_image.GetPixelID())

        # Registration
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

        # Connect all of the observers so that we can perform plotting during registration.
        # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
        # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

        final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                      sitk.Cast(moving_image, sitk.sitkFloat32))

        # print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
        # print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

        moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                         moving_image.GetPixelID())

        temp_output_dir = os.path.join(output_dir, *moving_dir.split('/')[-2:])
        if not os.path.exists(temp_output_dir):
            os.makedirs(temp_output_dir)
        output_path = os.path.join(temp_output_dir, moving_all_mod[i].split('/')[-1])

        sitk.WriteImage(moving_resampled, output_path)

    # Prefer SITK's WriteImage and ReadImage to directly passing the Numpy data to the calling function
    return os.path.join(*moving_dir.split('/')[-2:])

# moving = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/MICCAI_BraTS_2018_Data_Training/Nifti/HGG/Brats18_2013_2_1/Brats18_2013_2_1_t1.nii.gz'
# fixed = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/atlas_resampled_oriented.nii'
# output_dir = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/brats_registered.nii'
# register(moving=moving, fixed=fixed, output_dir=output_dir)
