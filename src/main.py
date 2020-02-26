import os
import sys
import argparse
import tensorflow as tf
from signal_transformation import helpers, tf_transformation
from src.models import resnet_34, resnet_50
import src.metrics as metrics


def parse_args():
    """Parse arguments.

        Args:

        Returns:

        Raises:

    """
    parser = argparse.ArgumentParser(description='The app allows to different ResNet architectures')

    # Required argument
    parser.add_argument('-t', action='store_true', help='Train or not a model')
    parser.add_argument('-a', type=str, help='Type of architectures: resnet_34, resnet_50')
    parser.add_argument('-o', type=str, help='Output directory')
    parser.add_argument('-m', type=str, help='Path to a pre-trained model')

    parser.add_argument('--input-dev', type=str, help='Input directory with wav files for train')
    parser.add_argument('--input-eval', type=str,
                        help='Input directory with wav files for evaluation')

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


def train(model, dev_out_dir, eval_out_dir):
    train_dataset = tf.data.TFRecordDataset(
        filenames=[item for item in helpers.find_files(dev_out_dir, pattern=['.tfrecords'])]
    )
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the spectrograms and labels.
    train_dataset = train_dataset.shuffle(buffer_size=300)
    train_dataset = train_dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Randomizes input using a window of 256 elements (read into memory)
    train_dataset = train_dataset.repeat(10)  # Repeats dataset this # times
    train_dataset = train_dataset.batch(120)  # Batch size to use
    # dataset = dataset.prefetch(3)

    eval_dataset = tf.data.TFRecordDataset(
        filenames=[item for item in helpers.find_files(eval_out_dir, pattern=['.tfrecords'])]
    )
    eval_dataset = eval_dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    eval_dataset = eval_dataset.batch(120)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy', metrics.eer]
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir='./logs/resnet'
    )

    history = model.fit(
        train_dataset,
        epochs=2,
        steps_per_epoch=100,
        callbacks=[tensorboard_callback],
        verbose=1
    )

    model.evaluate(eval_dataset)

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
        # helpers.create_dir(dev_out_dir)
        # tf_transformation.wav_to_tf_records(
        #     audio_path=args.input_dev,
        #     out_path=dev_out_dir,
        #     size=5000,
        #     spec_format=tf_transformation.SpecFormat.STFT,
        #     spec_shape=(300, 80, 1)
        # )

        eval_out_dir = os.path.join(args.o, 'eval')
        helpers.create_dir(eval_out_dir)
        tf_transformation.wav_to_tf_records(
            audio_path=args.input_eval,
            out_path=eval_out_dir,
            spec_format=tf_transformation.SpecFormat.MEL_SPEC,
            spec_shape=(300, 80, 1)
        )

        model = train(model, dev_out_dir, eval_out_dir)

    else:
        model = tf.keras.models.load_model(args.m)

    sys.exit()


if __name__ == "__main__":
    main()
