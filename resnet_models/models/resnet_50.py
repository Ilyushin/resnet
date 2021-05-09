import tensorflow as tf
from resnet_models.models import blocks


def get_model(input_shape=(300, 80, 1), weight_decay=1e-4, n_classes=5994):
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

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        n_classes,
        kernel_initializer='orthogonal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        name='fc1000')(
        x)

    # A softmax that is followed by the models loss must be done cannot be done
    # in float16 due to numeric issues. So we pass dtype=float32.
    x = tf.keras.layers.Activation('softmax', dtype='float32')(x)

    # Create models
    model = tf.keras.models.Model(inputs=X_input, outputs=x, name='ResNet50')

    return model
