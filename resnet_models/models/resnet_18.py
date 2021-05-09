import tensorflow as tf
from resnet_models.models import blocks


def get_model(input_shape=None, embeddings_size=512, weight_decay=1e-4, n_classes=5994):
    # Define the input as a tensor with shape input_shape
    x_input = tf.keras.layers.Input(input_shape)

    # Zero-Padding
    x = tf.keras.layers.ZeroPadding2D((3, 3))(x_input)

    # Stage 1
    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(7, 7),
        strides=(2, 2), name='conv1',
        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0),
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = blocks.resnet_18_basic_block_layer(
        input_tensor=x,
        filters=64,
        num_blocks=2,
        stride=1
    )

    # Stage 3 (≈2 lines)
    x = blocks.resnet_18_basic_block_layer(
        input_tensor=x,
        filters=128,
        num_blocks=2,
        stride=2
    )

    # Stage 4 (≈6 lines)
    x = blocks.resnet_18_basic_block_layer(
        input_tensor=x,
        filters=256,
        num_blocks=2,
        stride=2
    )

    # Stage 5 (≈3 lines)
    x = blocks.resnet_18_basic_block_layer(
        input_tensor=x,
        filters=512,
        num_blocks=2,
        stride=2
    )

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        n_classes,
        kernel_initializer='orthogonal',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        name='fc1000')(
        x
    )

    # A softmax that is followed by the models loss must be done cannot be done
    # in float16 due to numeric issues. So we pass dtype=float32.
    x = tf.keras.layers.Activation('softmax', dtype='float32')(x)

    # Create models
    model = tf.keras.models.Model(inputs=x_input, outputs=x, name='ResNet18')

    return model
