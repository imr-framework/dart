import os
import pickle

import numpy as np
import tqdm

from dart.train import models

# =========
# LOAD Y - REPORT FEATURE VECTORS
# =========
path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/Girish_reports'
with open(os.path.join(path, 'report_feature_vector.p'), 'rb') as p:
    report_feature_vector = pickle.load(p)
subject_names = report_feature_vector['Subject']  # IDs of BraTS subjects
tumour_type_flags = np.array(report_feature_vector['Tumour type'])
edema_flags = np.array(report_feature_vector['Edema'])
mass_effect_flags = np.array(report_feature_vector['Mass effect'])

# =========
# LOAD X - FEATURE VECTORS
# =========
dart_feature_vector = []
dart_feature_vector_path = '/Users/sravan953/Documents/CU/Projects/imr-framework/DART/Data/DART_feature_vec'
for i, s in enumerate(subject_names):
    tumour_type = 'HGG' if tumour_type_flags[i] == 1 else 'LGG'
    load_path = os.path.join(dart_feature_vector_path, tumour_type, s) + '.npy'
    dart_feature_vector.append(np.load(load_path))
dart_feature_vector = np.array(dart_feature_vector)
tum_vox = dart_feature_vector[:, -3:]
dart_feature_vector = dart_feature_vector[:, :-3]


def leave_one_out_training(model, y):
    acc = []
    for i in tqdm.tqdm(range(50)):
        test_idx = np.random.random_integers(0, high=len(dart_feature_vector) - 1, size=4)
        train_idx = np.arange(len(dart_feature_vector))
        train_idx = np.delete(train_idx, test_idx)
        x_train = dart_feature_vector[train_idx]
        x_test = dart_feature_vector[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(x=x_train, y=y_train, epochs=100, verbose=0)
        acc.append(model.evaluate(x=x_test, y=y_test)[1])
    print(np.mean(acc))
    return model


# =========
# TRAIN
# =========
x_train = dart_feature_vector
model = models.get_tumour_type_mass_effect_model()
y_train = np.vstack((tumour_type_flags, mass_effect_flags)).T
# model = models.get_edema_model()
# y_train = edema_flags
model = leave_one_out_training(model, y_train)
# model.save('../test/tumour_type_mass_effect_model.hdf5')
# model.save('../test/edema_model.hdf5')
