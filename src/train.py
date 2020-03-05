import os
import tensorflow as tf
import tensorflow_addons as tfa
from signal_transformation import helpers
import src.metrics as metrics
from src.settings import MAIN
from src.data_generator import DataGenerator


def get_data(path_to_files):
    x = []
    y = {}
    labels = {}
    counter = 0
    for idx, file_path in enumerate(helpers.find_files(path_to_files, pattern=['.npy'])):
        x.append((idx, file_path))
        speaker_id = file_path.split('/')[-3]
        if speaker_id not in labels.keys():
            labels[speaker_id] = counter
            counter += 1

        y[idx] = labels[speaker_id]

    return x, y


def train(model, dev_out_dir, valid_out_dir, number_dev_files=0, number_val_files=0, epochs=100,
          batch_size=128):
    # Parameters
    params = {
        'dim': MAIN['shape'],
        'batch_size': batch_size,
        'n_classes': MAIN['n_classes'],
        'n_channels': 1,
        'shuffle': True
    }

    # Datasets
    train_files, train_labels = get_data(dev_out_dir)
    valid_files, valid_labels = get_data(valid_out_dir)

    # Generators
    training_generator = DataGenerator(train_files, train_labels, **params)
    validation_generator = DataGenerator(valid_files, valid_labels, **params)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss='categorical_crossentropy',
        metrics=['acc', metrics.eer]
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='./logs/resnet/tensorboard/'
    )

    helpers.create_dir('./logs/resnet/checkpoints/')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='./logs/resnet/checkpoints/model.{epoch:02d}.tf',
        verbose=0,
        save_weights_only=False,
        save_freq='epoch',
        save_best_only=True,
        monitor='val_eer'
    )

    steps_per_epoch = int(number_dev_files / batch_size)
    validation_steps = int(number_val_files / batch_size)
    print('Started train the model')
    model.fit(
        training_generator,
        validation_data=validation_generator,
        use_multiprocessing=False,
        workers=1,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tensorboard_callback, cp_callback],
        validation_steps=validation_steps,
        verbose=1
    )
    print('Finished train the model')

    # history_eval = model.evaluate(valid_dataset, use_multiprocessing=True, verbose=0)

    # print('Eval loss:', history_eval[0])
    # print('Eval err:', history_eval[1])

    return model
