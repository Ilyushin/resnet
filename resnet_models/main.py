import os
import sys
import argparse
import tensorflow as tf
from signal_transformation import helpers, tf_transformation
from resnet_models.models import resnet_34, resnet_50
from resnet_models.train import train
from resnet_models.settings import MAIN

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
    parser.add_argument('-t', action='store_true', help='Train or not a models.')
    parser.add_argument('-a', type=str, help='Type of architectures: resnet_34, resnet_50.')
    parser.add_argument('-o', type=str, help='Output directory.')
    parser.add_argument('-p', action='store_true', help='Preparing or not data.')
    parser.add_argument('--pretrained-models', type=str, help='Path to a pre-trained models.')
    parser.add_argument('--save-models', type=str, help='Path to a place fro saving a models.')
    parser.add_argument('--input-dev', type=str, help='Input directory with wav files for train.')
    parser.add_argument('--input-eval', type=str,
                        help='Input directory with wav files for evaluation.')
    parser.add_argument('-e', type=int, default=100, help='Number of epochs.')
    parser.add_argument('-b', type=int, default=128, help='Butch size.')
    parser.add_argument('--format', type=str,
                        help='Type of data: pcm, stft, mel_spec, log_mel_spec, mfcc')

    return parser.parse_args()


def main():
    """Entry point.

            Args:

            Returns:

            Raises:

        """

    args = parse_args()
    model = None
    shape = MAIN['shape']

    if args.t:
        if args.a == 'resnet_34':
            model = resnet_34.get_model(input_shape=shape,
                                        embeddings_size=512,
                                        n_classes=MAIN['n_classes'])
        elif args.a == 'resnet_50':
            model = resnet_50.get_model()
        else:
            print('Need to specify the architecture.')
            sys.exit()

        data_dir = os.path.join(args.o, 'data/')
        dev_out_dir = os.path.join(data_dir, 'dev/')
        valid_out_dir = os.path.join(data_dir, 'eval/')

        number_dev_files = len(
            [item for item in helpers.find_files(args.input_dev, pattern=['.wav'])])
        number_val_files = len(
            [item for item in helpers.find_files(args.input_eval, pattern=['.wav'])])

        if args.p:
            spect_format = tf_transformation.SpecFormat.PCM
            if args.format == 'stft':
                spect_format = tf_transformation.SpecFormat.STFT
            elif args.format == 'mel_spec':
                spect_format = tf_transformation.SpecFormat.MEL_SPEC
            elif args.format == 'log_mel_spec':
                spect_format = tf_transformation.SpecFormat.LOG_MEL_SPEC
            elif args.format == 'mfcc':
                spect_format = tf_transformation.SpecFormat.MFCC

            print('Started preparing train data')
            helpers.create_dir(dev_out_dir)
            tf_transformation.wav_to_numpy_arrays(
                audio_path=args.input_dev,
                out_path=dev_out_dir,
                spec_format=spect_format,
                spec_shape=shape
            )
            print('Finished preparing train data')
            print()
            print('Started preparing validation data')
            helpers.create_dir(valid_out_dir)
            tf_transformation.wav_to_numpy_arrays(
                audio_path=args.input_eval,
                out_path=valid_out_dir,
                spec_format=spect_format,
                spec_shape=shape
            )
            print()
            print('Finished preparing validation data')

        # print(models.summary())

        model = train(
            model,
            dev_out_dir,
            valid_out_dir,
            args.o,
            number_dev_files=number_dev_files,
            number_val_files=number_val_files,
            epochs=args.e,
            batch_size=args.b
        )

        if args.save_model:
            path_to_model = os.path.join(args.save_model, '{}.h5'.format(args.a))
            helpers.create_dir(args.save_model)
            model.save(path_to_model, save_format='tf')

    else:
        model = tf.keras.models.load_model(args.pretrained_model)

    sys.exit()


if __name__ == "__main__":
    main()
