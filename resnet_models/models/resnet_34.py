import tensorflow as tf
from resnet_models.models import blocks


def get_model(input_shape=(257, 998, 1), embeddings_size=512, weight_decay=1e-4, n_classes=5994):
    # Define the input as a tensor with shape input_shape
    input_layer = tf.keras.layers.Input(shape=input_shape, name='input')

    # Zero-Padding
    # x = tf.keras.layers.ZeroPadding2D((3, 3))(input_layer)

    # Stage 1
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        padding='same',
        use_bias=False,
        name='conv1',
        kernel_initializer='orthogonal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
    )(input_layer)
    x = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Stage 2
    x = blocks.convolutional_block(x, kernel_size=3, filters=[48, 48, 96], stage=2, block='a',
                                   strides=(1, 1), trainable=True)
    x = blocks.identity_block(x, 3, [48, 48, 96], stage=2, block='b', trainable=True)

    # Stage 3 (≈4 lines)
    x = blocks.convolutional_block(x, kernel_size=3, filters=[96, 96, 128], stage=3, block='a',
                                   trainable=True)
    x = blocks.identity_block(x, 3, [96, 96, 128], stage=3, block='b', trainable=True)
    x = blocks.identity_block(x, 3, [96, 96, 128], stage=3, block='c', trainable=True)

    # Stage 4 (≈6 lines)
    x = blocks.convolutional_block(x, kernel_size=3, filters=[128, 128, 256], stage=4, block='a',
                                   trainable=True)
    x = blocks.identity_block(x, 3, [128, 128, 256], stage=4, block='b', trainable=True)
    x = blocks.identity_block(x, 3, [128, 128, 256], stage=4, block='c', trainable=True)

    # Stage 5 (≈3 lines)
    x = blocks.convolutional_block(x, kernel_size=3, filters=[256, 256, 512], stage=5, block='a',
                                   trainable=True)
    x = blocks.identity_block(x, 3, [256, 256, 512], stage=5, block='b', trainable=True)
    x = blocks.identity_block(x, 3, [256, 256, 512], stage=5, block='c', trainable=True)

    x = tf.keras.layers.MaxPooling2D(
        pool_size=(3, 1),
        strides=(2, 1),
        padding='same'
    )(x)

    # x = tf.keras.layers.Conv2D(
    #     filters=embeddings_size,
    #     kernel_size=(7, 1),
    #     strides=(1, 1),
    #     activation='relu',
    #     kernel_initializer='orthogonal',
    #     use_bias=True,
    #     trainable=True,
    #     kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
    #     bias_regularizer=tf.keras.regularizers.l2(weight_decay),
    #     name='x_fc')(x)

    # x = tf.keras.layers.AveragePooling2D((1, 5), strides=(1, 1), name='avg_pool')(x)
    # x = tf.keras.layers.Reshape((-1, embeddings_size), name='reshape')(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(
        embeddings_size,
        activation=None,
        kernel_initializer='orthogonal',
        use_bias=True,
        trainable=True,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        name='embedding')(x)

    x = tf.keras.layers.Dense(
        n_classes,
        # activation='softmax',
        name='fc' + str(embeddings_size),
        kernel_initializer='orthogonal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        # bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        use_bias=False,
        trainable=True
    )(x)

    y = tf.keras.layers.Activation('softmax', name='predictions')(x)

    # y = tf.nn.l2_normalize(x, axis=1, epsilon=1e-12, name='output')

    # Create models
    model = tf.keras.models.Model(inputs=input_layer, outputs=y, name='ResNet34')

    return model
