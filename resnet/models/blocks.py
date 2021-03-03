import tensorflow as tf


def convolutional_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),
                        weight_decay=1e-4,
                        trainable=True):
    """
    Arguments:
    x -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    strides -- Integer, specifying the stride to be used

    Returns:
    x -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    filter1, filter2, filter3 = filters

    dropout = 0.5

    ##### MAIN PATH #####
    # First component of main path
    x = tf.keras.layers.Conv2D(
        filters=filter1,
        kernel_size=(1, 1),
        strides=strides,
        use_bias=False,
        trainable=trainable,
        name=conv_name_base + '2a',
        kernel_initializer='orthogonal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Second component of main path
    x = tf.keras.layers.Conv2D(
        filters=filter2,
        kernel_size=kernel_size,
        padding='same',
        use_bias=False,
        trainable=trainable,
        name=conv_name_base + '2b',
        kernel_initializer='orthogonal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.Dropout(dropout)(x)

    # Third component of main path
    x = tf.keras.layers.Conv2D(
        filters=filter3,
        kernel_size=(1, 1),
        use_bias=False,
        trainable=trainable,
        name=conv_name_base + '2c',
        kernel_initializer='orthogonal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name=bn_name_base + '2c')(x)

    ##### SHORTCUT PATH ####
    x_shortcut = tf.keras.layers.Conv2D(
        filters=filter3,
        kernel_size=(1, 1),
        strides=strides,
        use_bias=False,
        trainable=trainable,
        name=conv_name_base + '1',
        kernel_initializer='orthogonal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(input_tensor)
    x_shortcut = tf.keras.layers.BatchNormalization(
        axis=3,
        trainable=trainable,
        name=bn_name_base + '1'
    )(x_shortcut)
    # x = tf.keras.layers.Dropout(dropout)(x)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    x = tf.keras.layers.Add()([x, x_shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def identity_block(input_tensor, kernel_size, filters, stage, block, weight_decay=1e-4,
                   trainable=True):
    """
    Arguments:
    x -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    x -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    filter1, filter2, filter3 = filters

    # First component of main path
    x = tf.keras.layers.Conv2D(
        filters=filter1,
        kernel_size=(1, 1),
        use_bias=False,
        trainable=trainable,
        name=conv_name_base + '2a',
        kernel_initializer='orthogonal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Second component of main path
    x = tf.keras.layers.Conv2D(
        filters=filter2,
        kernel_size=kernel_size,
        padding='same',
        use_bias=False,
        trainable=trainable,
        name=conv_name_base + '2b',
        kernel_initializer='orthogonal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    # Third component of main path
    x = tf.keras.layers.Conv2D(
        filters=filter3,
        kernel_size=(1, 1),
        use_bias=False,
        trainable=trainable,
        name=conv_name_base + '2c',
        kernel_initializer='orthogonal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=3, trainable=trainable, name=bn_name_base + '2c')(x)

    # Final step
    x = tf.keras.layers.Add()([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)

    return x


class VladPooling(tf.keras.layers.Layer):
    '''
    This layer follows the NetVlad, GhostVlad
    '''

    def __init__(self, mode, k_centers, g_centers=0, **kwargs):
        self.k_centers = k_centers
        self.g_centers = g_centers
        self.mode = mode
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.cluster = self.add_weight(
            shape=[self.k_centers + self.g_centers, input_shape[0][-1]],
            name='centers',
            initializer='orthogonal'
        )
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape
        return (input_shape[0][0], self.k_centers * input_shape[0][-1])

    def call(self, x):
        # feat : bz x W x H x D, cluster_score: bz X W x H x clusters.
        feat, cluster_score = x
        num_features = feat.shape[-1]

        # softmax normalization to get soft-assignment.
        # A : bz x W x H x clusters
        max_cluster_score = tf.keras.backend.max(cluster_score, -1, keepdims=True)
        exp_cluster_score = tf.keras.backend.exp(cluster_score - max_cluster_score)
        A = exp_cluster_score / K.sum(exp_cluster_score, axis=-1, keepdims=True)

        # Now, need to compute the residual, self.cluster: clusters x D
        A = tf.keras.backend.expand_dims(A, -1)  # A : bz x W x H x clusters x 1
        feat_broadcast = tf.keras.backend.expand_dims(
            feat,
            -2)  # feat_broadcast : bz x W x H x 1 x D
        feat_res = feat_broadcast - self.cluster  # feat_res : bz x W x H x clusters x D
        weighted_res = tf.multiply(A, feat_res)  # weighted_res : bz x W x H x clusters x D
        cluster_res = tf.keras.backend.sum(weighted_res, [1, 2])

        if self.mode == 'gvlad':
            cluster_res = cluster_res[:, :self.k_centers, :]

        cluster_l2 = tf.keras.backend.l2_normalize(cluster_res, -1)
        outputs = tf.keras.backend.reshape(
            cluster_l2,
            [-1, int(self.k_centers) * int(num_features)]
        )

        return outputs


def resnet_18_basic_block(input_tensor=None, filters=None, stride=None):
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=(3, 3),
                               strides=stride,
                               kernel_initializer='he_normal',
                               padding="same")(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=(3, 3),
                               strides=1,
                               kernel_initializer='he_normal',
                               padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    shortcut = input_tensor
    if stride != 1:
        shortcut = tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=(1, 1),
                                          strides=stride,
                                          kernel_initializer='he_normal')(input_tensor)
        shortcut = tf.keras.layers.BatchNormalization(axis=3)(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def resnet_18_basic_block_layer(input_tensor=None, filters=None, num_blocks=None, stride=1):
    x = resnet_18_basic_block(input_tensor=input_tensor, filters=filters, stride=stride)

    for _ in range(1, num_blocks):
        x = resnet_18_basic_block(input_tensor=input_tensor, filters=filters, stride=stride)

    return x


