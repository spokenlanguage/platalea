import json
import logging
import numpy as np
import pathlib
import torch

import platalea.dataset as D
import platalea.rank_eval as E
from utils.extract_transcriptions import extract_trn
from platalea.experiments.config import get_argument_parser


args = get_argument_parser()
logging.basicConfig(level=logging.INFO)

# Parsing arguments
args.add_argument(
    'path', metavar='path', help='Model\'s path')
args.add_argument(
    '--asr_model_dir',
    help='Path to the directory where the pretrained ASR/SLT model is stored',
    dest='asr_model_dir', type=str, action='store')
args.add_argument(
    '-t', help='Evaluate on test set', dest='use_test_set',
    action='store_true', default=False)
args.add_argument(
    '--text_image_model_dir',
    help='Path to the directory where the pretrained text-image model is \
    stored',
    dest='text_image_model_dir', type=str, action='store')
args.enable_help()
args.parse()


batch_size = 16

logging.info('Loading data')
if args.use_test_set:
    data = D.flickr8k_loader(split='test', batch_size=batch_size,
                             shuffle=False)
else:
    data = D.flickr8k_loader(split='val', batch_size=batch_size,
                             shuffle=False)
logging.info('Loading ASR/SLT model')
if args.asr_model_dir:
    path = pathlib.Path(args.asr_model_dir) / 'net.best.pt'
else:
    path = pathlib.Path(args.path) / 'asr.best.pt'
net = torch.load(path)
logging.info('Extracting ASR/SLT transcriptions')
with torch.no_grad():
    net.eval()
    hyp_asr, _ = extract_trn(net, data.dataset, use_beam_decoding=True)
logging.info('Loading text-image model')
if args.text_image_model_dir:
    path = pathlib.Path(args.text_image_model_dir) / 'net.best.pt'
else:
    path = pathlib.Path(args.path) / 'ti.best.pt'
net = torch.load(path)
logging.info('Evaluating text-image with ASR/SLT\'s output')
with torch.no_grad():
    net.eval()
    data = data.dataset.evaluation()
    correct = data['correct'].cpu().numpy()
    image_e = net.embed_image(data['image'])
    text_e = net.embed_text(hyp_asr)
    result = E.ranking(image_e, text_e, correct)
    res_out = dict(medr=np.median(result['ranks']),
                   recall={1: np.mean(result['recall'][1]),
                           5: np.mean(result['recall'][5]),
                           10: np.mean(result['recall'][10])})
print(json.dumps(res_out))
