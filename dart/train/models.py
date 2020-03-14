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


def get_necrosis_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(input_shape=(129,), units=256, activation='relu'),
        tf.keras.layers.Dense( units=64, activation='relu'),
        tf.keras.layers.Dense(units=4, activation='sigmoid'),
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def get_report_model():
    img_input = tf.keras.layers.Input(shape=(240, 240, 155, 1))
    conv3d1 = tf.keras.layers.Conv3D(filters=32, kernel_size=3, activation='relu')(img_input)
    gap1 = tf.keras.layers.GlobalAveragePooling3D()(conv3d1)
    # conv3d2 = tf.keras.layers.Conv3D(filters=16, kernel_size=3, activation='relu')(conv3d1)
    # conv3d3 = tf.keras.layers.Conv3D(filters=8, kernel_size=3, activation='relu')(conv3d2)
    flatten = tf.keras.layers.Flatten()(gap1)

    vec_input = tf.keras.layers.Input(shape=(129,))
    dense1 = tf.keras.layers.Dense(units=256, activation='relu')(vec_input)

    concatenate = tf.keras.layers.Concatenate(axis=-1)([flatten, dense1])

    dense2 = tf.keras.layers.Dense(units=128, activation='relu')(concatenate)
    dense3 = tf.keras.layers.Dense(units=12, activation='softmax')(dense2)

    model = tf.keras.models.Model(inputs=[img_input, vec_input], outputs=[dense3])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
