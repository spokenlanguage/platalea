import pathlib
import configargparse

home = pathlib.Path.home()
default_config_files = [str(home / '.platalea/config.ini'),
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

args.add_argument('-c', '--config', is_config_file=True, help='config file path')
args.add_argument('--data_root', env_var='PLATALEA_DATA_ROOT',
                  action='store', default='/roaming/gchrupal/datasets/flickr8k/',
                  help='location of the flickr8k (or similar) dataset')
args.add_argument('--meta', env_var='PLATALEA_METADATA_JSON',
                  action='store', default='dataset_multilingual.json',
                  help='filename of the metadata file (dataset.json or similar) relative to the dataset location')
args.add_argument('--audio_features_fn', env_var='PLATALEA_AUDIO_FEATURES_FN',
                  action='store', default='mfcc_delta_features.pt',
                  help='filename of the MFCC audio features file relative to the dataset location')
args.add_argument('--audio_subdir', env_var='PLATALEA_AUDIO_SUBDIR',
                  action='store', default='flickr_audio/wavs/',
                  help='directory containing the flickr8k wav files, relative to the dataset location')
args.add_argument('--image_subdir', env_var='PLATALEA_IMAGE_SUBDIR',
                  action='store', default='Flickr8k_Dataset/Flicker8k_Dataset/',
                  help='directory containing the flickr8k image files, relative to the dataset location')
args.add_argument('--language', env_var='PLATALEA_LANGUAGE',
                  action='store', default='en',
                  help='language to use for the transcriptions/translations')
args.add_argument('--seed', default=123, type=int,
                  help='seed for sources of randomness')
