import nibabel as nb
import numpy as np

from dart.train import models


def leave_one_out_training(model, x_train, y_train):
    arr_acc_training = []
    arr_acc_validation = []
    for i in range(len(x_train)):
        print(f'{i + 1}/{len(x_train)}')
        _idx_train = np.ones(len(x_train), dtype=np.bool)
        _idx_train[i] = False
        _x_train = x_train[_idx_train]
        _y_train = y_train[_idx_train]
        _x_validation = np.expand_dims(x_train[i], axis=0)
        _y_validation = np.expand_dims(y_train[i], axis=0)

        history = model.fit(x=_x_train, y=_y_train, validation_data=(_x_validation, _y_validation), epochs=100,
                            verbose=0)

        acc_training = history.history['acc']
        acc_validation = history.history['val_acc']

        arr_acc_training.append(acc_training)
        arr_acc_validation.append(acc_validation)
    return model, arr_acc_training, arr_acc_validation


def necrosis_etC():
    x_vec_path = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\4MICCAI_2020\DART_x_vec_output.npy"
    y_vec_path = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\4MICCAI_2020\DART_y_vec_output.npy"
    X = np.load(x_vec_path)[:, :-3]
    Y = np.load(y_vec_path)[:, [0, -3, -2, -1]]
    # Y = np.load(y_vec_path)[:, 0]

    idx_train = np.random.choice(np.arange(len(X)), size=len(X) - 5, replace=False)
    x_train = X[idx_train]
    y_train = Y[idx_train]
    x_test = np.delete(X, idx_train, axis=0)
    y_test = np.delete(Y, idx_train, axis=0)

    # Choose model
    model = models.get_necrosis_model()  # <-- CHANGE ME
    model, arr_acc_training, arr_acc_validation = leave_one_out_training(model, x_train, y_train)

    acc_training = np.mean(arr_acc_training)
    acc_validation = np.mean(arr_acc_validation)
    acc_testing = model.evaluate(x=x_test, y=y_test, verbose=0)[1]  # Loss and metric are returned
    print(f'Training/validation/testing accuracy: {acc_training}/{acc_validation}/{acc_testing}')

    # model.save('../test/tumour_type_mass_effect_model.hdf5')
    # model.save('../test/edema_model.hdf5')


def report():
    x_vec_path = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\4MICCAI_2020\DART_x_vec_output.npy"
    y_vec_path = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\4MICCAI_2020\DART_y_vec_output.npy"
    all_paths_path = r"C:\Users\sravan953\Documents\CU\Projects\imr-framework\DART\Data\4MICCAI_2020\DART_paths_output.npy"
    X = np.load(x_vec_path)[:, :-3]  # Last three values are tumor volumes
    Y = np.load(y_vec_path)[:, 1:4]
    paths = np.load(all_paths_path)

    idx_train = np.random.choice(np.arange(len(X)), size=len(X) - 5, replace=False)
    x_train = X[idx_train]
    y_train = Y[idx_train]
    paths_train = paths[idx_train]
    volumes_train = [nb.load(p).get_fdata() for p in paths_train]
    volumes_train = np.asarray(volumes_train)
    volumes_train = np.expand_dims(volumes_train, axis=-1)
    x_test = np.delete(X, idx_train, axis=0)
    y_test = np.delete(Y, idx_train, axis=0)
    paths_test = np.delete(paths, idx_train, axis=0)
    # volumes_test = [nb.load(p).get_fdata() for p in paths_test]

    # Choose model
    model = models.get_report_model()  # <-- CHANGE ME
    model.fit([volumes_train, x_train], y_train, epochs=10)
    # model, arr_acc_training, arr_acc_validation = leave_one_out_training(model, x_train, y_train)
    # op = model.predict(x_test)

    # acc_training = np.mean(arr_acc_training)
    # acc_validation = np.mean(arr_acc_validation)
    # acc_testing = model.evaluate(x=x_test, y=y_test, verbose=0)[1]  # Loss and metric are returned
    # print(f'Training/validation/testing accuracy: {acc_training}/{acc_validation}/{acc_testing}')

    # model.save('../test/tumour_type_mass_effect_model.hdf5')
    # model.save('../test/edema_model.hdf5')


necrosis_etC()
