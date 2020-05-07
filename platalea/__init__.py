import configargparse
import os
from pathlib import Path

home = Path.home()
# Looking at the home folder, then at current folder
dflt_conf_files = [
    home / '.platalea/config.ini',
    home / '.platalea/config.yml',
    'config.ini',
    'config.yml']
configargparse.init_argument_parser(
    name='platalea', default_config_files=dflt_conf_files)
parser = configargparse.get_argument_parser(name='platalea')
parser.add_argument(
    '-c', '--config', is_config_file=True, help='config file path')
parser.add_argument(
    '--audio_features_fn', env_var='PLATALEA_AUDIO_FEATURES_FN',
    default='mfcc_delta_features.pt',
    help='filename of the MFCC audio features file relative to the dataset \
    location')

# Flickr8k specific parameters
parser.add_argument(
    '--flickr8k_root', env_var='FLICKR8K_ROOT',
    default='/roaming/gchrupal/datasets/flickr8k/',
    help='location of the flickr8k dataset')
parser.add_argument(
    '--flickr8k_meta', env_var='FLICKR8K_METADATA_JSON',
    default='dataset_multilingual.json', dest='flickr8k_meta',
    help='filename of the metadata file (dataset.json or similar) relative to \
    the dataset location')
parser.add_argument(
    '--flickr8k_audio_subdir', env_var='FLICKR8K_AUDIO_SUBDIR',
    default='flickr_audio/wavs/',
    help='directory containing the flickr8k wav files, relative to the \
    dataset location')
parser.add_argument(
    '--flickr8k_image_subdir', env_var='FLICKR8K_IMAGE_SUBDIR',
    default='Flickr8k_Dataset/Flicker8k_Dataset/',
    help='directory containing the flickr8k image files, relative to the \
    dataset location')
parser.add_argument(
    '--flickr8k_language', env_var='FLICKR8K_LANGUAGE', default='en',
    help='language to use for the transcriptions/translations')

# Librispeech specific parameters
parser.add_argument(
    '--librispeech_root', env_var='LIBRISPEECH_ROOT',
    default='/home/bjrhigy/corpora/LibriSpeech',
    help='location of the librispeech dataset')
parser.add_argument(
    '--librispeech_meta', env_var='librispeech_METADATA_JSON',
    default='metadata.json', dest='librispeech_meta',
    help='filename of the metadata file (metadata.json or similar) relative to \
    the dataset location')
