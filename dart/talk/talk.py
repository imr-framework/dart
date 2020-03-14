import multiprocessing
import os

import nibabel as nb
import numpy as np

from dart.talk.google_tts import tts_main
from utils import make_freesurfer_dict
from utils.plot3d_multiproc import multiproc_plot3d
from talk.regex import is_word_in_list

"""
TUMOUR REGIONS:
1 - tumour core (non-enhancing)
2 - whole tumour
4 - enhancing tumour (necrotic)
"""

TUMOUR_GRADE = 'HGG'
SUBJECT = 'Brats18_2013_10_1'

# =========
# PATHS
# =========
BASE_DATA_PATH = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data'
ATLAS_PATH = os.path.join(BASE_DATA_PATH, '4ISMRM_2020', 'atlas_seg.mgz')
BRAIN_PATH = os.path.join(BASE_DATA_PATH, 'MICCAI_BraTS_2018_Data_Training', TUMOUR_GRADE, SUBJECT,
                          f'{SUBJECT}_###.nii.gz')
DART_FEATURE_VECTOR = os.path.join(BASE_DATA_PATH, '4ISMRM_2020', 'DART_feature_vec', TUMOUR_GRADE, f'{SUBJECT}.npy')
DART_TUMOUR_TYPE_PRED = os.path.join(BASE_DATA_PATH, '4ISMRM_2020', 'DART_tumour_type_pred', TUMOUR_GRADE,
                                     f'{SUBJECT}.npy')
DART_MASS_EFFECT_PRED = os.path.join(BASE_DATA_PATH, '4ISMRM_2020', 'DART_mass_effect_pred', TUMOUR_GRADE,
                                     f'{SUBJECT}.npy')
DART_EDEMA_PRED = os.path.join(BASE_DATA_PATH, '4ISMRM_2020', 'DART_edema_pred', TUMOUR_GRADE, f'{SUBJECT}.npy')
SEG_MASK_PATH = os.path.join(BASE_DATA_PATH, '4ISMRM_2020', 'DART_Nifti_seg', TUMOUR_GRADE, f'{SUBJECT}.nii.gz')
# =========

TUMOR_REGIONS = ['Core', 'Whole', 'Enhancing']


def answer(text):
    print(f'[A] {text}')
    tts_main(text)


def setup():
    # =========
    # DICTIONARIES
    # =========
    edema_dict = {0: 'No edema', 1: 'Mild edema', 2: 'Moderate edema', 3: 'Extensive edema'}
    mass_effect_dict = {0: 'No mass effect', 1: 'Yes'}
    tumour_type_dict = {0: 'LGG', 1: 'HGG'}

    # =========
    # GENERATE PREDICTIONS
    # =========
    feature_vec = np.load(DART_FEATURE_VECTOR)
    print(f'Features: {feature_vec.shape}')
    feature_vec = np.expand_dims(feature_vec, axis=0)
    tumour_type_pred = np.load(DART_TUMOUR_TYPE_PRED)[0]
    mass_effect_pred = np.load(DART_MASS_EFFECT_PRED)[0]
    edema_pred = np.load(DART_EDEMA_PRED)[0]
    tumour_type_flag = tumour_type_dict[tumour_type_pred]
    mass_effect_flag = mass_effect_dict[mass_effect_pred]
    edema_flag = edema_dict[edema_pred]

    return {'feature_vec': feature_vec, 'tumour_type': tumour_type_flag, 'mass_effect': mass_effect_flag,
            'edema': edema_flag}


def main():
    params = setup()
    feature_vec = params['feature_vec']

    # answer('Welcome to DART Talk.')
    # answer("You can tell DART what to do by starting your commands with 'DART'.")
    command = None
    while command != -1:
        # command = str.lower(stt_main())
        # command = command.split(' ')
        command = ['dart', 'proportion']

        if 'dart' in command:
            if 'show' in command:
                if 'data' in command or 'tumor' in command or 'image' in command:
                    if 't1' in command:
                        _brain_path = BRAIN_PATH.replace('###', 't1')
                    elif 'contrast' in command:
                        _brain_path = BRAIN_PATH.replace('###', 't1ce')
                    elif 't2' in command:
                        _brain_path = BRAIN_PATH.replace('###', 't2')
                    elif 'flair' in command:
                        _brain_path = BRAIN_PATH.replace('###', 'flair')
                    else:
                        answer("I'm sorry, I didn't get that.")
                        continue
                    vol = nb.load(_brain_path).get_fdata()

                    # Start Matplotlib plotting in a separate process
                    vol_q = multiprocessing.Queue()
                    mask_q = multiprocessing.Queue()
                    p = multiprocessing.Process(target=multiproc_plot3d, args=(vol_q, mask_q))
                    vol_q.put(vol)
                    mask_q.put(None)
                    p.start()
                    answer('Okay, here you go...')
                if 'mask' in command:
                    mask = nb.load(SEG_MASK_PATH).get_fdata()
                    if 'whole' in command:
                        mask[np.where(mask != 2)] = 0
                    elif 'core' in command:
                        mask[np.where(mask != 1)] = 0
                    elif 'enhancing' in command:
                        mask[np.where(mask != 4)] = 0
                    mask_q.put(mask)
                    answer('Okay, here you go...')
                else:
                    answer("I'm sorry, I didn't get that.")
            elif 'volume' in command:
                if 'whole' in command:
                    # What is the whole tumour volume?
                    answer(f'Tumour volume: {int(feature_vec[0, -2])} mm x mm x mm')
                elif 'core' in command:
                    # What is the tumour core volume?
                    answer(f'Tumour volume: {int(feature_vec[0, -3])} mm x mm x mm')
                elif 'enhancing' in command:
                    # What is the enhancing tumour volume?
                    answer(f'Tumour volume: {int(feature_vec[0, -1])} mm x mm x mm')
            elif is_word_in_list(term=['mass', 'effect'], sentence=command):
                # Is there mass effect?
                mass_effect = params['mass_effect']
                answer(f'Mass effect: {mass_effect}')
            elif 'edema' in command:
                # Describe the edema.
                edema = params['edema']
                answer(f'Edema: {edema}')
            elif is_word_in_list(term=['type', 'tumor'], sentence=command):
                # What is the tumour type?
                tumour_type = params['tumour_type']
                answer(f'Tumour type: {tumour_type}')
            elif is_word_in_list(term=['multifocal', 'regions'], sentence=command):
                freesurfer_labels = make_freesurfer_dict.main()
                brats_seg = nb.load(SEG_MASK_PATH).get_fdata()
                atlas_seg = nb.load(ATLAS_PATH).get_fdata()
                idx1 = np.where(brats_seg == 1)  # Core tumor
                regions1 = np.unique(atlas_seg[idx1])
                regions1 = [freesurfer_labels[x] for x in regions1]
                idx2 = np.where(brats_seg == 2)  # Whole tumor
                regions2 = np.unique(atlas_seg[idx2])
                regions2 = [freesurfer_labels[x] for x in regions2]
                idx4 = np.where(brats_seg == 4)  # Enhancing tumor
                regions4 = np.unique(atlas_seg[idx4])
                regions4 = [freesurfer_labels[x] for x in regions4]
                regions = list(set(regions1).union(regions2, regions4))
                answer(f'The tumour occupies {len(regions)} regions.')
                print(f'The tumour occupies the following {len(regions)} regions: {regions}')
            elif is_word_in_list(term=['intensity', 'intensities'], sentence=command):
                # h_flags is as follows: [[tumour region 1 for flair, t1, t1ce, t2][region 2...][region 4...]]
                h_flags = feature_vec[:, [42, 43, 44, 45, 89, 90, 91, 92, 136, 137, 138, 139]]
                h_flags = h_flags.reshape((3, 4))
                contrasts = ['FLAIR', 'T1', 'T1ce', 'T2']
                intensity_dict = {0: 'hypointense', 1: 'hyperintense'}
                for counter1, tr in enumerate(TUMOR_REGIONS):
                    print(f'== Tumour region: {tr} ==')
                    for counter2, c in enumerate(contrasts):
                        _flag = intensity_dict[h_flags[counter1, counter2]]
                        print(f'{c}: {_flag}')
                answer('Here are the intensity flags for each of the tumour regions across the four M.R. contrasts.')
            elif is_word_in_list(term='proportion', sentence=command):
                """
                - Tumour proportions is a flattened array of length 144
                - Discard last 3 elements: three tumour region volumes
                - Reshape from (1, 141) to (3, 47)
                - Discard last (3, 4) corresponding to intensity flags of the three tumour regions in the four contrasts
                - Resulting shape is (3, 44) 
                """
                tum_proportions = feature_vec[0, :-3].reshape((3, -1))[:, :-4]
                freesurfer_labels = make_freesurfer_dict.main()
                max_tum_proportions = np.argsort(tum_proportions, axis=1)[:, -3:].flatten()
                # min_tum_proportions = np.argsort(tum_proportions, axis=1)[:, :3]
                max_tum_proportions_mapped = [freesurfer_labels[i] for i in max_tum_proportions]
                max_tum_proportions_mapped = np.array(max_tum_proportions_mapped).reshape((3, -1))
                for counter, tr in enumerate(TUMOR_REGIONS):
                    print(f'The {tr.lower()} tumor occurs most widely in: {max_tum_proportions_mapped[counter]}')
                answer('The tumor most widely occurs in the listed regions.')
            else:
                answer("I'm sorry, I didn't get that.")


if __name__ == '__main__':
    main()
