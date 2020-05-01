import configargparse
import logging
import numpy as np
import pickle
import random
import os
from shutil import copyfile
import torch

import platalea.asr as M1
import platalea.dataset as D
import platalea.rank_eval as E
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
parser.add_argument(
    '--text_image_model_dir',
    help='Path to the directory where the pretrained text-image model is \
    stored',
    dest='text_image_model_dir', type=str, action='store')
config_args, _ = parser.parse_known_args()

# Setting general configuration
torch.manual_seed(config_args.seed)
random.seed(config_args.seed)
logging.basicConfig(level=logging.INFO)


batch_size = 8

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(split='train', batch_size=batch_size, shuffle=True),
    val=D.flickr8k_loader(split='val', batch_size=batch_size, shuffle=False))

if config_args.asr_model_dir:
    net = torch.load(os.path.join(config_args.asr_model_dir, 'net.best.pt'))
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
hyp_asr, _ = extract_trn(net, data['val'].dataset, use_beam_decoding=True)

if config_args.text_image_model_dir:
    net = torch.load(os.path.join(config_args.text_image_model_dir,
                                  'net.best.pt'))
else:
    logging.info('Building model text-image')
    net = M2.TextImage(M2.get_default_config())
    run_config = dict(max_lr=2 * 1e-4, epochs=32)
    logging.info('Training text-image')
    M2.experiment(net, data, run_config)
    copyfile('result.json', 'result_text_image.json')
    copy_best('result_text_image.json', 'ti.best.pt')
    net = torch.load('ti.best.pt')

logging.info('Evaluating text-image with ASR\'s output')
data = data['val'].dataset.evaluation()
correct = data['correct'].cpu().numpy()
image_e = net.embed_image(data['image'])
text_e = net.embed_text(hyp_asr)
result = E.ranking(image_e, text_e, correct)
print(dict(medr=np.median(result['ranks']),
           recall={1: np.mean(result['recall'][1]),
                   5: np.mean(result['recall'][5]),
                   10: np.mean(result['recall'][10])}))
