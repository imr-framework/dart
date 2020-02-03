import os

import nibabel as nb
import numpy as np

from dart.train import register_elastix
# import dart.anisoconv_seg as anisoconv_seg

# =========
# PATHS
# =========
BASE_DATA_PATH = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data'

ATLAS_PATH = os.path.join(BASE_DATA_PATH, 'atlas.nii')
ATLAS_SEG_PATH = os.path.join(BASE_DATA_PATH, 'atlas_seg.mgz')
BASE_CODE_PATH = os.path.dirname(__file__)
BRATS_CONFIG_FILE_PATH = os.path.join(BASE_CODE_PATH, 'anisoconv_config', 'config.txt')
BRATS_DATA_PATH = os.path.join(BASE_DATA_PATH, 'MICCAI_BraTS_2018_Data_Training')
BRATS_TEST_FILE_PATH = os.path.join(BASE_CODE_PATH, 'anisoconv_config', 'test.txt')
DART_OUTPUT_REGISTERED = os.path.join(BASE_DATA_PATH, 'DART_Nifti_reg')
DART_OUTPUT_SEGMENTED = os.path.join(BASE_DATA_PATH, 'DART_Nifti_seg')
DART_OUTPUT_FEATURE_VECTORS = os.path.join(BASE_DATA_PATH, 'DART_feature_vec')


def __get_brats_reg_of_current_subject(brats_subject_type: str, brats_subject_name: str):
    """
    1. Read test.txt file to identify BraTS subject in focus.
    2. Load the registered BraTS Nifti file from disk.
    """
    brats_subject_full_path = os.path.join(DART_OUTPUT_REGISTERED, brats_subject_type, brats_subject_name)
    files = sorted(os.listdir(brats_subject_full_path))
    brats_reg = []
    for f in files:
        if 'seg' not in f:
            brats_reg.append(nb.load(os.path.join(brats_subject_full_path, f)).get_fdata())

    return brats_reg


def compute_tumour_relative_intensities(brats_subject_type: str, brats_subject_name: str, brats_seg):
    brats_reg = __get_brats_reg_of_current_subject(brats_subject_type, brats_subject_name)

    h_flags = []  # Hyper/hypo flags for each tumour t
    for t in [1, 2, 4]:  # Tumour regions are 1, 2 and 4
        temp_h_flags = []
        for b in brats_reg:
            idx = np.where(brats_seg == t)  # Coordinates of tumour t in BraTS
            mean_intensity_tum = np.mean(b[idx])  # Mean intensity of tumour t
            temp_b = b
            temp_b[idx] = 0  # Set tumour indices to 0 to get 'remaining brain volume'
            temp_b = temp_b[np.nonzero(temp_b)]
            mean_intensity_remaining_brain = np.mean(temp_b)  # Mean intensity of remaining brain volume
            temp_h_flags.append(1 if mean_intensity_remaining_brain - mean_intensity_tum < 0 else 0)
        h_flags.append(temp_h_flags)

    return h_flags


def compute_tumour_proportions_of_anat(atlas_seg, brats_seg):
    tum_proportions = []
    tum_vox = []  # Number of voxels of tumour t

    for i, t in enumerate([1, 2, 4]):  # Tumour regions are 1, 2 and 4
        # Step 1: Construct tumour t and overlay with corresponding anatomy labels from atlas_seg
        idx = np.where(brats_seg == t)  # Coordinates of tumour t in BraTS
        tum_vol = np.zeros_like(brats_seg)
        tum_vol[idx] = atlas_seg[idx]  # Now tum_vol is anatomy labels corresponding to tumour t
        tum_labels, tum_labels_counts = np.unique(tum_vol,
                                                  return_counts=True)  # Anatomy labels in tumour t and corresponding counts
        tum_labels = tum_labels[1:]  # Remove background label 0
        tum_labels_counts = tum_labels_counts[1:]  # Remove background label 0

        # Step 2: Compute atlas label counts S.T. all labels are present in tumour t
        atlas_labels, atlas_labels_counts = np.unique(atlas_seg,
                                                      return_counts=True)  # All anatomy labels and corresponding counts
        atlas_labels = atlas_labels[1:]  # Remove background label 0
        atlas_labels_counts = atlas_labels_counts[1:]  # Remove background label 0
        common_labels, _, idx_in_atlas = np.intersect1d(tum_labels, atlas_labels, assume_unique=True,
                                                        return_indices=True)  # Filter labels not present in tumour t
        common_labels_counts = np.take(atlas_labels_counts, idx_in_atlas)

        # Step 3: Compute proportion of tumour t in BraTS subject brats
        p = np.divide(tum_labels_counts, common_labels_counts)
        p2 = np.zeros(len(atlas_labels))
        p2[idx_in_atlas] = p

        tum_proportions.append(p2)
        tum_vox.append(len(idx[0]))

    return tum_proportions, tum_vox


# def get_one_brats_subject() -> str:
#     # Iterate through entire BraTS dataset
#     brats_files = []
#     for x in os.walk(BRATS_DATA_PATH):
#         parent, children_folders, children_items = x
#         if len(children_folders) == 0:
#             brats_files.append(parent)
#
#     for i, x in enumerate(brats_files):
#         print(f'=== SUBJECT {i + 1} ===')
#         yield x

def get_one_brats_subject() -> str:
    all_files = []
    GIRISH_REPORTS = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/Girish_reports/docx'
    all_idx = []
    for f in os.listdir(GIRISH_REPORTS):
        f = f.replace('brats_', '')
        f = f.replace('.docx', '')
        all_idx.append(f)

    # Iterate through BraTS dataset matching reports
    brats_files = []
    for parent, children_folders, children_items in os.walk(BRATS_DATA_PATH):
        if len(children_folders) == 0:
            folder = parent.split('/')[-1]
            id = folder.split('_')[-2]
            if id in all_idx:
                brats_files.append(parent)

    for i, x in enumerate(sorted(brats_files)):
        print(f'=== SUBJECT {i + 1} ===')
        yield x


def main():
    # =========
    # LOADING
    # =========
    # DO NOT load the atlas, elastix registration takes care of that; assign variable to maintain convention
    atlas = ATLAS_PATH
    atlas_seg = nb.load(ATLAS_SEG_PATH).get_fdata()
    for brats_subject_full_path in get_one_brats_subject():
        temp = brats_subject_full_path.split(os.sep)[-2:]
        brats_subject_type, brats_subject_name = temp[0], temp[1]

        # Register BraTS to atlas
        # print('Registering to atlas...')
        # reg_output_paths = register_elastix.register(moving_dir=brats_subject_full_path, fixed=atlas,
        #                                              output_dir=DART_OUTPUT_REGISTERED)
        #
        # # Save BraTS registered images to disk
        # print('Writing registered files to test.txt for anisoconv segmentation...')
        # with open(BRATS_TEST_FILE_PATH, 'w') as f:  # Write to test.txt for anisoconv segmentation
        #     f.write(reg_output_paths)
        #
        # # Segment using anisotropic conv network
        # print('Segmenting...')
        # brats_seg = anisoconv_seg.test(config_file=BRATS_CONFIG_FILE_PATH)
        brats_seg = nb.load(os.path.join(DART_OUTPUT_SEGMENTED, brats_subject_type, brats_subject_name + '.nii.gz'))
        brats_seg = brats_seg.get_fdata()

        # Compute percentage of vol qualifying for each label
        print('Computing tumour proportions and performing volumetry...')
        tum_proportions, tum_vox = compute_tumour_proportions_of_anat(atlas_seg=atlas_seg, brats_seg=brats_seg)

        # Compute relative intensities of each tumour region t
        print('Computing hypo/hyper flags...')
        h_flags = compute_tumour_relative_intensities(brats_subject_type, brats_subject_name, brats_seg)

        feature_vector = np.concatenate((tum_proportions, h_flags), axis=1)  # Feature vector is now 43 + 4
        feature_vector = feature_vector.flatten()  # Feature vector is now 47 * 3
        feature_vector = np.concatenate(
            (feature_vector, tum_vox))  # Add this at the end because we will not use it for DL
        np.save(os.path.join(DART_OUTPUT_FEATURE_VECTORS, brats_subject_type, brats_subject_name + '.npy'),
                feature_vector)


main()
