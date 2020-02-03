import numpy as np
import skimage.transform


def fix_orientation(vol):
    # Run this first
    fixed_vol = np.transpose(vol, axes=(0, 2, 1))
    fixed_vol = np.rot90(fixed_vol, k=2, axes=(0, 1))

    return fixed_vol


def resample_vol(vol, target_shape=(240, 240, 155)):
    # Run this second
    resampled_v = np.zeros(target_shape)

    temp_v = np.zeros((target_shape[0], target_shape[1], vol.shape[2]))
    for i in range(vol.shape[2]):
        slice = vol[:, :, i]
        temp_v[:, :, i] = skimage.transform.resize(slice, target_shape[:-1])

    for i in range(temp_v.shape[0]):
        slice = vol[i]
        resampled_v[i] = skimage.transform.resize(slice, target_shape[1:])

    return resampled_v
