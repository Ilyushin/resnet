"""
Tests for resnet/models/resnet_18.py
"""

import tensorflow as tf
from resnet.models import resnet_18


def test_resnet_18():
    model = resnet_18.get_model(
        input_shape=(150, 150, 3),
        embeddings_size=512,
        weight_decay=1e-4,
        n_classes=5994
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss='categorical_crossentropy',
        metrics=['acc']
    )
