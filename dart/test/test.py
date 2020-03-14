import os
import pickle

import numpy as np
import tensorflow as tf

# =========
# LOAD TEST DATA
# =========
# Load report feature vector only to get list of subjects
print('Loading data...')
path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/Girish_reports'
with open(os.path.join(path, 'report_feature_vector.p'), 'rb') as p:
    report_feature_vector = pickle.load(p)
subject_names = report_feature_vector['Subject']  # IDs of BraTS subjects
tumour_type_flags = np.array(report_feature_vector['Tumour type'])

dart_feature_vector = []
dart_feature_vector_path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/DART_feature_vec'
for i, s in enumerate(subject_names):
    tumour_type = 'HGG' if tumour_type_flags[i] == 1 else 'LGG'
    load_path = os.path.join(dart_feature_vector_path, tumour_type, s) + '.npy'
    dart_feature_vector.append(np.load(load_path))
dart_feature_vector = np.array(dart_feature_vector)

num_test_cases = 50
# random_idx = np.random.random_integers(low=0, high=len(dart_feature_vector) - 1, size=num_test_cases)
random_idx = np.arange(num_test_cases)
subject_names = np.take(subject_names, random_idx)
dart_feature_vector = dart_feature_vector[random_idx]
tum_vox = dart_feature_vector[:, -3:]  # Extract tumour volumes
# Extract hypo/hyper flags
h_flags = dart_feature_vector[:, [42, 43, 44, 45, 89, 90, 91, 92, 136, 137, 138, 139]]
h_flags = h_flags.reshape((num_test_cases, 3, -1))  # 3 tumour types
dart_feature_vector = dart_feature_vector[:, :-3]  # Now take only the feature vector

# =========
# LOAD MODELS
# =========
print('Predicting...')
tumour_type_mass_effect_model = tf.keras.models.load_model('tumour_type_mass_effect_model.hdf5')
edema_model = tf.keras.models.load_model('edema_model.hdf5')
y_pred = [tumour_type_mass_effect_model.predict(dart_feature_vector),
          edema_model.predict_classes(dart_feature_vector)]

# =========
# GENERATE REPORTS
# =========
print('Generating reports...')
REPORT_SAVE_PATH = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/DART_reports'

tumour_type_flags = np.round(y_pred[0][:, 0])
mass_effect_flags = np.round(y_pred[0][:, 1])
edema_flags = y_pred[1]

edema_dict = {0: 'No edema', 1: 'Mild edema', 2: 'Moderate edema', 3: 'Extensive edema'}
mass_effect_dict = {0: 'no mass effect', 1: 'mass effect'}
tumour_type_dict = {0: 'LGG', 1: 'HGG'}

for i in range(num_test_cases):
    case_no = subject_names[i].split('_')[-2]
    h_flag = h_flags[i]
    edema_flag = edema_flags[i]
    mass_effect_flag = mass_effect_flags[i]
    tumour_type_flag = tumour_type_flags[i]
    tum_vol = tum_vox[i]

    report = ['MRI BRAIN SCAN',
              '=========',
              'Technique: T1, T1 contrast enhanced, T2 and FLAIR axial sequences',
              f'Case: {case_no}',
              f'Tumour type: {tumour_type_dict[tumour_type_flag]}']
    for row, t in enumerate(['Non-enhancing tumour', 'Whole tumour', 'Necrotic tumour']):
        report.append(f'\n{t}')
        report.append(f'Volume: {tum_vol[row]} mm3')
        for col, c in enumerate(['FLAIR', 'T1', 'T1 contrast enhanced', 'T2']):
            v = h_flag[row, col]
            if v == 0:
                v = 'Hypointense'
            elif v > 0 and v < 0.5:
                v = 'Predominantly hypointense'
            elif v > 0.5 and v < 1:
                v = 'Predominantly hyperintense'
            elif v == 1:
                v = 'Hyperintense'
            report.append(f'{c}: {v}')

    report.append(f'\nEdema: {edema_dict[edema_flag]} with {mass_effect_dict[mass_effect_flag]}')
    report = '\n'.join(report)

    with open(os.path.join(REPORT_SAVE_PATH, subject_names[i]) + '.txt', 'w') as f:
        f.write(report)
print('Done.')
