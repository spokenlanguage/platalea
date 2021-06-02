import configargparse
import logging
from pathlib import Path

home = Path.home()
# Looking at the home folder, then at current folder (later ones in list override previous ones)
default_config_files = [
    str(home / '.platalea/config.ini'),
    str(home / '.platalea/config.yml'),
    'config.ini',
    'config.yml']


class PlataleaConfig:
    """
    Help message is disabled by default, but can be reenabled using enable_help().
    This makes it possible to parse multiple times, e.g. to add arguments from
    multiple places without triggering help by parses in between.
    """
    def __init__(self):
        self._args = {}
        self._parser = configargparse.ArgParser(default_config_files=default_config_files,
                                                add_help=False, formatter_class=configargparse.DefaultsFormatter)
        self._help = False

    def add_argument(self, *args, **kwargs):
        arg = self._parser.add_argument(*args, **kwargs)
        self._args[arg.dest] = arg.default

    def enable_help(self):
        self._help = True

    def parse(self):
        parsed_args, unknown = self._parser.parse_known_args()
        if "-h" in unknown or "--help" in unknown:
            if self._help:
                self._parser.print_help()
                exit(0)
        for arg, value in vars(parsed_args).items():
            self._args[arg] = value
        if self._args['verbose']:
            logging.basicConfig(level=logging.DEBUG)
        elif self._args['silent']:
            logging.basicConfig(level=logging.WARNING)
        else:
            logging.basicConfig(level=logging.INFO)

    def __getattr__(self, arg_name):
        return self._args[arg_name]

    def __str__(self):
        return self._args.__str__()


def get_argument_parser():
    args = PlataleaConfig()

    args.add_argument(
        '-c', '--config', is_config_file=True, help='config file path')
    args.add_argument(
        "-v", "--verbose", help="increase output verbosity",
        action="store_true")
    args.add_argument(
        "--silent", help="decrease output verbosity", action="store_true")
    args.add_argument(
        '--audio_features_fn', env_var='PLATALEA_AUDIO_FEATURES_FN',
        default='mfcc_features.pt',
        help='filename of the MFCC audio features file relative to the dataset \
        location')
    args.add_argument(
        '--seed', default=123, type=int, help='seed for sources of randomness')
    args.add_argument(
        '--epochs', env_var='PLATALEA_EPOCHS', action='store', default=32,
        type=int, help='number of epochs after which to stop training')
    args.add_argument(
        '--downsampling-factor', default=None, type=float,
        dest='downsampling_factor',
        help='factor by which the dataset should be downsampled')
    args.add_argument(
        '--optimizer', default='adam',
        choices=['adam', 'adadelta'],
        help='The optimizer used for the learning process.')
    args.add_argument(
        '--lr_scheduler', default="cyclic",
        choices=['cyclic', 'noam', 'constant'],
        help='The learning rate scheduler to use. WARNING: noam not yet \
        implemented for most experiments!')
    args.add_argument(
        '--cyclic_lr_max', default=2 * 1e-4, type=float,
        help='Maximum learning rate for cyclic learning rate scheduler')
    args.add_argument(
        '--cyclic_lr_min', default=1e-6, type=float,
        help='Minimum learning rate for cyclic learning rate scheduler')
    args.add_argument(
        '--constant_lr', default=1e-4, type=float,
        help='Learning rate for constant learning rate scheduler')
    args.add_argument(
        '--device', type=str, default=None, env_var="PLATALEA_DEVICE",
        help="Device to train on. Can be passed on to platalea.hardware.device \
        in experiments.")
    args.add_argument(
        '--hidden_size_factor', type=int, default=1024,
        help='The experiment models by default have a factor 1024 in their \
        hidden size layers. With this parameter you can change that. For \
        testing purposes only!')
    args.add_argument(
        '--l2_regularization', default=0, type=float,
        help='L2 regularization using weight decay in the optimizer.')
    args.add_argument(
        '--loss_logging_interval', type=int, default=100,
        help='Step interval at which the training loss is logged on the info level.')
    args.add_argument(
        '--validation_interval', type=int, default=400,
        help='Step interval at which a validation step is run and logged on the info level.')

    # Flickr8k specific parameters
    args.add_argument(
        '--flickr8k_root', env_var='FLICKR8K_ROOT',
        default='/corpora/flickr8k/',
        help='location of the flickr8k dataset')
    args.add_argument(
        '--flickr8k_meta', env_var='FLICKR8K_METADATA_JSON',
        default='dataset.json',
        help='filename of the metadata file (dataset.json or similar) relative to \
        the dataset location')
    args.add_argument(
        '--flickr8k_audio_subdir', env_var='FLICKR8K_AUDIO_SUBDIR',
        default='flickr_audio/wavs/',
        help='directory containing the flickr8k wav files, relative to the \
        dataset location')
    args.add_argument(
        '--flickr8k_image_subdir', env_var='FLICKR8K_IMAGE_SUBDIR',
        default='Flicker8k_Dataset/',
        help='directory containing the flickr8k image files, relative to the \
        dataset location')
    args.add_argument(
        '--flickr8k_language', env_var='FLICKR8K_LANGUAGE', default='en',
        help='language to use for the transcriptions/translations')

    # HowTo100M
    args.add_argument(
        '--howto100m_root', env_var='HOWTO100M_ROOT',
        default='.',
        help='location of the howto100m dataset')
    args.add_argument(
        '--howto100m_audio_subdir', env_var='HOWTO100M_AUDIO_SUBDIR',
        default='audio',
        help='relative path of the audio subdir within the HowTo100M dir')
    args.add_argument(
        '--howto100m_video_subdir', env_var='HOWTO100M_VIDEO_SUBDIR',
        default='video',
        help='relative path of the video subdir within the HowTo100M dir')

    # Librispeech specific parameters
    args.add_argument(
        '--librispeech_root', env_var='LIBRISPEECH_ROOT',
        default='/home/bjrhigy/corpora/LibriSpeech',
        help='location of the librispeech dataset')
    args.add_argument(
        '--librispeech_meta', env_var='librispeech_METADATA_JSON',
        default='metadata.json',
        help='filename of the metadata file (metadata.json or similar) relative to \
        the dataset location')

    return args
