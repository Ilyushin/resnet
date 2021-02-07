import tensorflow as tf
from resnet.models import blocks


def get_model(input_shape=(300, 80, 1), embeddings_size=64):
    # Define the input as a tensor with shape input_shape
    X_input = tf.keras.layers.Input(input_shape)

    # Zero-Padding
    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(7, 7),
        strides=(2, 2), name='conv1',
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(X)
    X = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = blocks.convolutional_block(X, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', strides=1)
    X = blocks.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = blocks.identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = blocks.convolutional_block(X, kernel_size=3, filters=[128, 128, 512], stage=3, block='a', strides=2)
    X = blocks.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = blocks.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = blocks.identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = blocks.convolutional_block(X, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', strides=2)
    X = blocks.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = blocks.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = blocks.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = blocks.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = blocks.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = blocks.convolutional_block(X, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a', strides=2)
    X = blocks.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = blocks.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=2,
        padding='same'
    )(X)

    # output layer
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(
        embeddings_size, activation='softmax',
        name='fc' + str(embeddings_size),
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(X)

    X = tf.nn.l2_normalize(X, axis=1, epsilon=1e-12, name='output')

    # Create model
    model = tf.keras.models.Model(inputs=X_input, outputs=X, name='ResNet50')

    return model
