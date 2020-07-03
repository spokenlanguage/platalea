import configargparse
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
                                                add_help=False)
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

    def __getattr__(self, arg_name):
        return self._args[arg_name]


args = PlataleaConfig()

args.add_argument(
    '-c', '--config', is_config_file=True, help='config file path')
args.add_argument(
    '--audio_features_fn', env_var='PLATALEA_AUDIO_FEATURES_FN',
    default='mfcc_delta_features.pt',
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

# Flickr8k specific parameters
args.add_argument(
    '--flickr8k_root', env_var='FLICKR8K_ROOT',
    default='/roaming/gchrupal/datasets/flickr8k/',
    help='location of the flickr8k dataset')
args.add_argument(
    '--flickr8k_meta', env_var='FLICKR8K_METADATA_JSON',
    default='dataset_multilingual.json',
    help='filename of the metadata file (dataset.json or similar) relative to \
    the dataset location')
args.add_argument(
    '--flickr8k_audio_subdir', env_var='FLICKR8K_AUDIO_SUBDIR',
    default='flickr_audio/wavs/',
    help='directory containing the flickr8k wav files, relative to the \
    dataset location')
args.add_argument(
    '--flickr8k_image_subdir', env_var='FLICKR8K_IMAGE_SUBDIR',
    default='Flickr8k_Dataset/Flicker8k_Dataset/',
    help='directory containing the flickr8k image files, relative to the \
    dataset location')
args.add_argument(
    '--flickr8k_language', env_var='FLICKR8K_LANGUAGE', default='en',
    help='language to use for the transcriptions/translations')

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
