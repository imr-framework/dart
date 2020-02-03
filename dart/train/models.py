import tensorflow as tf


def get_tumour_type_mass_effect_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(input_shape=(141,), units=512, activation='relu'),
        tf.keras.layers.Dense(units=2, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_edema_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(input_shape=(141,), units=512, activation='relu'),
        tf.keras.layers.Dense(units=4, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

