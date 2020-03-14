from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd


def xlsx2yvec_cellwise():
    """
    Read Dr. Zenas' reports spreadsheet and convert to Y vector for training.
    The Y vector contains 15 features:
    1. Tumor type
    2. T1 - Intensity/Description/Location (3)
    2. T1ce - Intensity/Description/Location (3)
    3. T2 - Intensity/Description/Location (3)
    4. FLAIR - Intensity/Description/Location (3)
    5. Necrosis
    6. Satellite lesion
    """

    path_report = r'REPORT_ZENAS.xlsx'  # <-- CHANGE ME

    df = pd.read_excel(path_report)
    df = df.fillna('NA')
    arr_values = df.loc[:, 'T1 - Intensity':'FLAIR - Location'].values
    arr_values = np.concatenate(arr_values)
    arr_unique_values = np.unique(arr_values)
    dict_codes = dict(zip(arr_unique_values, np.arange(101, len(arr_unique_values) + 101)))
    dict_codes['N'] = 0
    dict_codes['LGG'] = 0
    dict_codes['Y'] = 1
    dict_codes['HGG'] = 1

    arr_y_vec = df.loc[:, 'Tumor type':'Mass effect'].applymap(lambda x: dict_codes[x])
    arr_subjects = df['Subject ID'].to_numpy()

    return arr_subjects, arr_y_vec


def compute_tumour_proportions_of_anat(atlas_seg, brats_seg):
    """
    TUMOUR REGIONS:
    1 - tumour core (non-enhancing)
    2 - whole tumour
    4 - enhancing tumour (necrotic)
    """

    tum_proportions = []
    tum_vox = []  # Number of voxels of tumour t

    for i, t in enumerate([1, 2, 4]):  # Tumour regions are 1 (tumour core), 2 (whole tumour) and 4 (enhancing tumour)
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

        x_vec = np.concatenate(tum_proportions)
        x_vec = np.concatenate((x_vec, tum_vox))

    return x_vec


def main():
    # =========
    # PATHS
    # =========
    path_base_data = Path(r'C:/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data')
    path_atlas_seg = path_base_data / '4MICCAI_2020' / 'atlas_labelled.mgz'
    path_brats_segmented = path_base_data / '4MICCAI_2020' / 'DART_Nifti_seg'
    path_x_vec_output = path_base_data / '4MICCAI_2020' / 'DART_x_vec_output.npy'
    path_y_vec_output = path_base_data / '4MICCAI_2020' / 'DART_y_vec_output.npy'
    path_paths_output = path_base_data / '4MICCAI_2020' / 'DART_paths.npy'

    atlas_seg = nb.load(path_atlas_seg).get_fdata()
    arr_all_brats_seg = list(path_brats_segmented.glob('*/*'))

    print('Constructing Y vector...')
    arr_subject_idx, arr_y_vec = xlsx2yvec_cellwise()  # Construct Y training vector
    print('Done.\n')

    arr_all_x_vec = []
    arr_all_y_vec = []
    arr_all_paths = []
    print('Constructing X vector...')
    for index, path_brats_seg_subject in enumerate(arr_all_brats_seg):
        print(f'{index + 1}/{len(arr_all_brats_seg)}...')
        subject_id = str(path_brats_seg_subject).split('_')[-2]
        y_vec = arr_y_vec.iloc[np.where(arr_subject_idx == int(subject_id))]
        y_vec = np.squeeze(y_vec.to_numpy())
        brats_seg = nb.load(str(path_brats_seg_subject)).get_fdata()
        x_vec = compute_tumour_proportions_of_anat(atlas_seg=atlas_seg, brats_seg=brats_seg)  # Takes time
        arr_all_x_vec.append(x_vec)
        arr_all_y_vec.append(y_vec)
        arr_all_paths.append(path_brats_seg_subject)
    print('Done.\n')

    print('Saving training vectors to disk...')
    arr_all_x_vec = np.stack(arr_all_x_vec)
    arr_all_y_vec = np.stack(arr_all_y_vec)
    arr_all_paths = [str(p) for p in arr_all_paths]
    arr_all_paths = np.stack(arr_all_paths)
    print(f'X: {arr_all_x_vec.shape}')
    print(f'Y: {arr_all_y_vec.shape}')
    print(f'Paths: {arr_all_paths.shape}')
    np.save(path_x_vec_output, arr_all_x_vec)
    np.save(path_y_vec_output, arr_all_y_vec)
    np.save(path_paths_output, arr_all_paths)
    print('Done.')


# main()
