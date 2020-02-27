import os
import sys
import argparse
import tensorflow as tf
from signal_transformation import helpers, tf_transformation
from src.models import resnet_34, resnet_50
from src.train import train


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
            path_to_model = os.path.join(args.save_model, '{}.h5'.format(args.a))
            helpers.create_dir(args.save_model)
            model.save(path_to_model, save_format='tf')

    else:
        model = tf.keras.models.load_model(args.pretrained_model)

    sys.exit()


if __name__ == "__main__":
    main()
