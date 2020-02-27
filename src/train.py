import os
import tensorflow as tf
import tensorflow_addons as tfa
from signal_transformation import helpers
import src.metrics as metrics


def parse_fn(serialized, spec_shape=(300, 80, 1)):
    size = spec_shape[0] * spec_shape[1] * spec_shape[2]
    features = {
        'spectrogram': tf.io.FixedLenFeature([size], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64)
    }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(
        serialized=serialized,
        features=features
    )

    spectrogram = tf.cast(parsed_example['spectrogram'], tf.float32)
    spectrogram = tf.reshape(spectrogram, [spec_shape[0], spec_shape[1], spec_shape[2]])
    label = tf.cast(parsed_example['label'], tf.int64)

    return spectrogram, label


def train(model, dev_out_dir, valid_out_dir, epochs=100, batch_size=128):
    train_files = [item for item in helpers.find_files(dev_out_dir, pattern=['.tfrecords'])]
    train_dataset = tf.data.TFRecordDataset(
        filenames=train_files
    )
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the spectrograms and labels.
    train_dataset = train_dataset.shuffle(buffer_size=300)
    train_dataset = train_dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Randomizes input using a window of 256 elements (read into memory)
    train_dataset = train_dataset.repeat()  # Repeats dataset this # times
    train_dataset = train_dataset.batch(batch_size)  # Batch size to use
    # dataset = dataset.prefetch(3)

    valid_files = [item for item in helpers.find_files(valid_out_dir, pattern=['.tfrecords'])]
    valid_dataset = tf.data.TFRecordDataset(
        filenames=valid_files
    )
    valid_dataset = valid_dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.repeat()
    valid_dataset = valid_dataset.batch(batch_size)

    model.compile(
        optimizer='adam',
        loss=tfa.losses.TripletSemiHardLoss(),
        metrics=[metrics.eer]
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

    print('Started train the model')
    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=int(len(train_files) / batch_size),
        callbacks=[tensorboard_callback, cp_callback],
        validation_data=valid_dataset,
        validation_steps=int(len(valid_files) / batch_size),
        verbose=1
    )
    print('Finished train the model')

    # history_eval = model.evaluate(valid_dataset, use_multiprocessing=True, verbose=0)

    # print('Eval loss:', history_eval[0])
    # print('Eval err:', history_eval[1])

    return model
