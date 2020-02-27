import tensorflow as tf


def convolutional_block(X, kernel_size, filters, stage, block, s=2):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    dropout = 0.5

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = tf.keras.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        strides=(s, s),
        name=conv_name_base + '2a',
        kernel_initializer=tf.keras.initializers.he_normal(seed=0),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Second component of main path
    X = tf.keras.layers.Conv2D(
        filters=F2,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name=conv_name_base + '2b',
        kernel_initializer=tf.keras.initializers.he_normal(seed=0),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.Dropout(dropout)(X)

    # Third component of main path
    X = tf.keras.layers.Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name=conv_name_base + '2c',
        kernel_initializer=tf.keras.initializers.he_normal(seed=0),
        kernel_regularizer = tf.keras.regularizers.l2(0.001)
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = tf.keras.layers.Conv2D(
        filters=F3,
        kernel_size=(1, 1),
        strides=(s, s),
        padding='valid',
        name=conv_name_base + '1',
        kernel_initializer=tf.keras.initializers.he_normal(seed=0),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(
        axis=3,
        name=bn_name_base + '1'
    )(X_shortcut)
    X = tf.keras.layers.Dropout(dropout)(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X


def identity_block(X, kernel_size, filters, stage, block):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = tf.keras.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name=conv_name_base + '2a',
        kernel_initializer=tf.keras.initializers.he_normal(seed=0),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Second component of main path
    X = tf.keras.layers.Conv2D(
        filters=F2,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='valid',
        name=conv_name_base + '2b',
        kernel_initializer=tf.keras.initializers.he_normal(seed=0),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = tf.keras.layers.Activation('relu')(X)

    # Third component of main path
    X = tf.keras.layers.Conv2D(
        filters=F3, kernel_size=(1, 1),
        strides=(1, 1), padding='valid',
        name=conv_name_base + '2c',
        kernel_initializer=tf.keras.initializers.he_normal(seed=0),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step
    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X
