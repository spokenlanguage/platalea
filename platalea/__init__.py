import configargparse

configargparse.init_argument_parser(name='platalea', default_config_files=['config.ini', 'config.yml'])
parser = configargparse.get_argument_parser(name='platalea')
parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
parser.add_argument('--data_root', env_var='PLATALEA_DATA_ROOT',
                    action='store', default='/roaming/gchrupal/datasets/flickr8k/',
                    dest='data_root',
                    help='location of the flickr8k (or similar) dataset')
parser.add_argument('--meta', env_var='PLATALEA_METADATA_JSON',
                    action='store', default='dataset_multilingual.json',
                    dest='meta',
                    help='filename of the metadata file (dataset.json or similar) relative to the dataset location')
parser.add_argument('--audio_features_fn', env_var='PLATALEA_AUDIO_FEATURES_FN',
                    action='store', default='mfcc_delta_features.pt',
                    dest='audio_features_fn',
                    help='filename of the MFCC audio features file relative to the dataset location')
parser.add_argument('--audio_subdir', env_var='PLATALEA_AUDIO_SUBDIR',
                    action='store', default='flickr_audio/wavs/',
                    dest='audio_subdir',
                    help='directory containing the flickr8k wav files, relative to the dataset location')
parser.add_argument('--image_subdir', env_var='PLATALEA_IMAGE_SUBDIR',
                    action='store', default='Flickr8k_Dataset/Flicker8k_Dataset/',
                    dest='image_subdir',
                    help='directory containing the flickr8k image files, relative to the dataset location')
