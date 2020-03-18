import configargparse
import os
from pathlib import Path

# Looking at the root of the repository, then at current folder
home = Path.home()
dflt_conf_files = [home / '.platalea/config.ini',
                   home / '.platalea/config.yml',
                   'config.ini',
                   'config.yml']
configargparse.init_argument_parser(name='platalea', default_config_files=dflt_conf_files)
parser = configargparse.get_argument_parser(name='platalea')
parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
parser.add_argument('--audio_features_fn', env_var='PLATALEA_AUDIO_FEATURES_FN',
                    action='store', default='mfcc_delta_features.pt',
                    dest='audio_features_fn',
                    help='filename of the MFCC audio features file relative to the dataset location')

# Flickr8k specific parameters
parser.add_argument('--flickr8k_root', env_var='FLICKR8K_ROOT',
                    action='store', default='/roaming/gchrupal/datasets/flickr8k/',
                    dest='flickr8k_root',
                    help='location of the flickr8k dataset')
parser.add_argument('--flickr8k_meta', env_var='FLICKR8K_METADATA_JSON',
                    action='store', default='dataset_multilingual.json',
                    dest='meta',
                    help='filename of the metadata file (dataset.json or similar) relative to the dataset location')
parser.add_argument('--flickr8k_audio_subdir', env_var='FLICKR8K_AUDIO_SUBDIR',
                    action='store', default='flickr_audio/wavs/',
                    dest='flickr8k_audio_subdir',
                    help='directory containing the flickr8k wav files, relative to the dataset location')
parser.add_argument('--flickr8k_image_subdir', env_var='FLICKR8K_IMAGE_SUBDIR',
                    action='store', default='Flickr8k_Dataset/Flicker8k_Dataset/',
                    dest='flickr8k_image_subdir',
                    help='directory containing the flickr8k image files, relative to the dataset location')
