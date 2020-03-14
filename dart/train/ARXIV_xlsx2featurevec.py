# import tensorflow.keras as keras

import re

import numpy as np
import pandas as pd

PATH_REPORT = r'REPORT_ZENAS.xlsx'  # <-- CHANGE ME
ARR_CODES = ['Acute-blood',
             'Anterior',
             'Along',
             'Basal',
             'Blood',
             'Central',
             'Enhancing',
             'Effacement',
             'Falx',
             'Frontal',
             'Ganglia',
             'Isointense core',
             'Hypointense',
             'Hyperintense',
             'Heterogeneous',
             'Lateral',
             'Left',
             'Midline shift',
             'Minimally-enhancing',
             'Multiple',
             'Necrosis',
             'Non-enhancing',
             'Occipital',
             'Parietal',
             'Peripheral',
             'Peripherally-enhancing',
             'Periventricular',
             'Partially-enhancing',
             'Right',
             'Rim-enhancing',
             'Scattered',
             'Surrounding',
             'Temporal',
             'Temporoparietal',
             'Vasogenic edema',
             'Ventricle']
REGEX_SPLIT_MULTIPLE = re.compile("\s*\(\d\)\s")  # Regex match template: (1) ABC (2) XYZ

df = pd.read_excel(PATH_REPORT)
df = df.fillna('NA')
SIZE_CODES = len(ARR_CODES)
DICT_CODIFY = dict(
    zip(ARR_CODES, np.arange(3, SIZE_CODES + 3)))  # We go from [2,SIZE_CODES) because 0 and 1 are for YES/NO
REGEX_MATCH_MULTIPLE = re.compile('\s*\(\d\)\s')


def _codify_location(arr_regex_splits):
    _location_code = []
    for location in arr_regex_splits:
        if location != '':
            _arr_location_tokenized = location.split(' ')
            # Capitalize each location token
            _arr_location_tokenized = list(map(str.capitalize, _arr_location_tokenized))
            # Remove occurrences of 'lobe' and 'junction'
            _arr_location_tokenized = list(filter(lambda x: x != 'Lobe' and x != 'Junction', _arr_location_tokenized))
            _sub_code = []
            for token in _arr_location_tokenized:
                if '/' in token:  # If location token has multiple anatomies
                    token = list(map(str.capitalize, token.split('/')))  # Split into sub-tokens
                    _sub_code.extend([DICT_CODIFY[t] for t in token])  # Codify each sub-token
                    _location_code.extend([DICT_CODIFY[t] for t in token])  # Codify each sub-token
                elif token != '':
                    # _sub_code.append(DICT_CODIFY[token])
                    _location_code.append(DICT_CODIFY[token])
        else:
            continue
        # _location_code.append(_sub_code)

    return _location_code


def _codify_intensity(arr_regex_splits):
    _intensity_code = []
    for intensity in arr_regex_splits:
        if intensity != '':
            _arr_intensity_tokenized = intensity.split(' ')
            # Capitalize each intensity token
            _arr_intensity_tokenized = list(map(str.capitalize, _arr_intensity_tokenized))
            for token in _arr_intensity_tokenized:
                if token != '':
                    _intensity_code.append(DICT_CODIFY[token])
        else:
            continue

    return _intensity_code


arr_codes = []
for index, value in df.iterrows():
    _arr_subject_codes = []
    for column in df.columns[2:-1]:  # T1, T1ce, T2 and FLAIR Intensity, Description and Location values only
        if 'Tumor type' in column or 'Necrosis' in column or 'Mass effect' in column or 'Satellite lesion' in column:
            v = value[column]
            _arr_subject_codes.append(2 if v == 'Y' else 1)
            continue
        arr_regex_splits = REGEX_MATCH_MULTIPLE.split(value[column])
        if 'Location' in column:
            _arr_subject_codes.extend(_codify_location(arr_regex_splits))
        elif 'Intensity' in column:
            _arr_subject_codes.extend(_codify_location(arr_regex_splits))
        else:
            arr_regex_splits = list(map(str.capitalize, arr_regex_splits))  # Capitalize each token
            _arr_subject_codes.extend([DICT_CODIFY[x] for x in arr_regex_splits if x != '' and x.upper() != 'NA'])
    arr_codes.append(_arr_subject_codes)
