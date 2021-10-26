import keras
import numpy as np
import pandas as pd
import util

np.random.seed(813306)


def build_block(input, n_feature_maps, drop_rate):
    print('build conv_x')

    conv_x = keras.layers.BatchNormalization()(input)
    conv_x = keras.layers.Conv2D(n_feature_maps, (8, 1), strides=(1, 1), padding='same',
                                 kernel_initializer="he_normal")(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    print('build conv_y')
    conv_y = keras.layers.Conv2D(n_feature_maps, (5, 1), strides=(1, 1), padding='same',
                                 kernel_initializer="he_normal")(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    print('build conv_z')
    conv_z = keras.layers.Conv2D(n_feature_maps, (3, 1), strides=(1, 1), padding='same',
                                 kernel_initializer="he_normal")(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input.shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv2D(n_feature_maps, (1, 1), strides=(1, 1), padding='same',
                                         kernel_initializer="he_normal")(input)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(input)
    print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)
    y = keras.layers.MaxPooling2D(pool_size=(4, 1), padding="same")(y)
    y = keras.layers.GaussianNoise(0.1)(y)
    if drop_rate != 0:
        y = keras.layers.Dropout(drop_rate)(y)

    return y


def build_resnet(input_shape, n_feature_maps, nb_classes):
    input = keras.layers.Input(shape=(input_shape), name='ecg')
    input_age = keras.layers.Input(shape=([1]), name='age')
    input_sex = keras.layers.Input(shape=([1]), name='sex')
    x = build_block(input, n_feature_maps, 0.2)
    x = build_block(x, 2 * n_feature_maps, 0.2)
    x = build_block(x, 2 * n_feature_maps, 0)

    full = keras.layers.GlobalAveragePooling2D()(x)
    full = keras.layers.Dropout(0.5)(full)
    full = keras.layers.Dense(4)(full)
    full = keras.layers.BatchNormalization()(full)
    full = keras.layers.Activation('relu')(full)
    age = keras.layers.BatchNormalization()(input_age)
    age = keras.layers.Activation('relu')(age)
    sex = keras.layers.BatchNormalization()(input_sex)
    sex = keras.layers.Activation('relu')(sex)
    full = keras.layers.concatenate([age, sex, full])
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    print('        -- model was built.')
    return [input, input_age, input_sex], out


def add_compile(model, **params):
    from keras.optimizers import Adam

    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy', util.auroc])


def build_network(**params):
    from keras.models import Model

    x, y = build_resnet(params['input_shape'], params['feature_maps'], params['num_categories'])

    model = Model(inputs=x, outputs=y)
    model.summary()
    add_compile(model, **params)
    return model