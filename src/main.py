import os
import sys
import argparse
import tensorflow as tf
import tensorflow_addons as tfa
from signal_transformation import helpers, tf_transformation
from src.models import resnet_34, resnet_50
import src.metrics as metrics

tf.config.experimental_run_functions_eagerly(True)


def parse_args():
    """Parse arguments.

        Args:

        Returns:

        Raises:

    """
    parser = argparse.ArgumentParser(
        description='The app allows to train different ResNet architectures')

    # Required argument
    parser.add_argument('-t', action='store_true', help='Train or not a model.')
    parser.add_argument('-a', type=str, help='Type of architectures: resnet_34, resnet_50.')
    parser.add_argument('-o', type=str, help='Output directory.')
    parser.add_argument('-p', action='store_true', help='Preparing or not data.')
    parser.add_argument('--pretrained-model', type=str, help='Path to a pre-trained model.')
    parser.add_argument('--save-model', type=str, help='Path to a place fro saving a model.')
    parser.add_argument('--input-dev', type=str, help='Input directory with wav files for train.')
    parser.add_argument('--input-eval', type=str,
                        help='Input directory with wav files for evaluation.')
    parser.add_argument('-e', type=int, default=100, help='Number of epochs.')
    parser.add_argument('-b', type=int, default=128, help='Butch size.')

    return parser.parse_args()


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

    valid_dataset = tf.data.TFRecordDataset(
        filenames=[item for item in helpers.find_files(valid_out_dir, pattern=['.tfrecords'])]
    )
    valid_dataset = valid_dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.batch(batch_size)

    model.compile(
        optimizer='adam',
        loss=tfa.losses.TripletSemiHardLoss(),
        metrics=[metrics.eer]
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='./logs/resnet/'
    )

    print('Started train the model')
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=int(len(train_files) / batch_size),
        callbacks=[tensorboard_callback],
        validation_data=valid_dataset,
        verbose=1
    )
    print('Finished train the model')

    return model


def main():
    """Entry point.

            Args:

            Returns:

            Raises:

        """

    args = parse_args()
    model = None

    if args.t:
        if args.a == 'resnet_34':
            model = resnet_34.get_model()
        elif args.a == 'resnet_50':
            model = resnet_50.get_model()
        else:
            print('Need to specify the architecture.')
            sys.exit()

        dev_out_dir = os.path.join(args.o, 'dev')
        valid_out_dir = os.path.join(args.o, 'eval')

        if args.p:
            print('Started preparing train data')
            helpers.create_dir(dev_out_dir)
            tf_transformation.wav_to_tf_records(
                audio_path=args.input_dev,
                out_path=dev_out_dir,
                spec_format=tf_transformation.SpecFormat.MEL_SPEC,
                spec_shape=(300, 80, 1)
            )
            print('Finished preparing train data')
            print()
            print('Started preparing validation data')
            helpers.create_dir(valid_out_dir)
            tf_transformation.wav_to_tf_records(
                audio_path=args.input_eval,
                out_path=valid_out_dir,
                spec_format=tf_transformation.SpecFormat.MEL_SPEC,
                spec_shape=(300, 80, 1)
            )
            print()
            print('Finished preparing validation data')

        model = train(model, dev_out_dir, valid_out_dir, epochs=args.e, batch_size=args.b)

        if args.save_model:
            model.save(os.path.join(args.save_model, 'model.h5'))

    else:
        model = tf.keras.models.load_model(args.pretrained_model)

    sys.exit()


if __name__ == "__main__":
    main()
