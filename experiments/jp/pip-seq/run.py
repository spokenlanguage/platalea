import configargparse
import logging
import pickle
import random
import os
from shutil import copyfile
import torch

import platalea.asr as M1
import platalea.dataset as D
import platalea.text_image as M2
from utils.copy_best import copy_best
from utils.extract_transcriptions import extract_trn

# Parsing arguments
parser = configargparse.get_argument_parser('platalea')
parser.add_argument(
    '--seed', default=123, type=int,
    help='seed for sources of randomness (default: 123)')
parser.add_argument(
    '--asr_model_dir',
    help='Path to the directory where the pretrained ASR model is stored',
    dest='asr_model_dir', type=str, action='store')
config_args, _ = parser.parse_known_args()

# Setting general configuration
torch.manual_seed(config_args.seed)
random.seed(config_args.seed)
logging.basicConfig(level=logging.INFO)


batch_size = 8

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(split='train', batch_size=batch_size, shuffle=True,
                            language='jp'),
    val=D.flickr8k_loader(split='val', batch_size=batch_size, shuffle=False,
                          language='jp'))
fd = D.Flickr8KData
if args.asr_model_dir:
    config_fpath = os.path.join(args.asr_model_dir, 'config.pkl')
    config = pickle.load(open(config_fpath, 'rb'))
    fd.le = config['label_encoder']
else:
    fd.init_vocabulary(data['train'].dataset)
    # Saving config
    pickle.dump(dict(label_encoder=fd.get_label_encoder(),
                     language='jp'),
                open('config.pkl', 'wb'))

if args.asr_model_dir:
    net = torch.load(os.path.join(args.asr_model_dir, 'net.best.pt'))
else:
    logging.info('Building ASR model')
    config = M1.get_default_config()
    net = M1.SpeechTranscriber(config)
    run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=32)
    logging.info('Training ASR')
    M1.experiment(net, data, run_config)
    copyfile('result.json', 'result_asr.json')
    copy_best('result_asr.json', 'asr.best.pt', experiment_type='asr')
    net = torch.load('asr.best.pt')

logging.info('Extracting ASR transcriptions')
for set_name in ['train', 'val']:
    ds = data[set_name].dataset
    hyp_asr, ref_asr = extract_trn(net, ds, use_beam_decoding=True)
    # Replacing original transcriptions with ASR's output
    for i in range(len(hyp_asr)):
        item = ds.split_data[i]
        if item[2] == ref_asr[i]:
            ds.split_data[i] = (item[0], item[1], hyp_asr[i])
        else:
            msg = 'Extracted reference #{} ({}) doesn\'t match dataset\'s \
                    one ({}) for {} set.'
            msg = msg.format(i, ref_asr[i], ds.split_data[i][3], set_name)
            logging.warning(msg)

if args.asr_model_dir:
    # Saving config for text-image model
    pickle.dump(dict(label_encoder=fd.get_label_encoder(),
                     language='en'),
                open('config.pkl', 'wb'))

logging.info('Building model text-image')
net = M2.TextImage(M2.get_default_config())
run_config = dict(max_lr=2 * 1e-4, epochs=32)

logging.info('Training text-image')
M2.experiment(net, data, run_config)
copyfile('result.json', 'result_text_image.json')
copy_best('result_text_image.json', 'ti.best.pt')
