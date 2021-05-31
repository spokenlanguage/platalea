import json
import logging
import numpy as np
import random
import os
from shutil import copyfile
import torch

import platalea.asr as M1
import platalea.dataset as D
import platalea.rank_eval as E
import platalea.text_image as M2
from platalea.utils.copy_best import copy_best
from platalea.utils.extract_transcriptions import extract_trn
from platalea.experiments.config import get_argument_parser


args = get_argument_parser()
# Parsing arguments
args.add_argument(
    '--asr_model_dir',
    help='Path to the directory where the pretrained ASR/SLT model is stored',
    dest='asr_model_dir', type=str, action='store')
args.add_argument(
    '--text_image_model_dir',
    help='Path to the directory where the pretrained text-image model is \
    stored',
    dest='text_image_model_dir', type=str, action='store')
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)

# Logging the arguments
logging.info('Arguments: {}'.format(args))


batch_size = 8

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
        args.audio_features_fn, split='train', batch_size=batch_size,
        shuffle=True, downsampling_factor=args.downsampling_factor),
    val=D.flickr8k_loader(
        args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
        args.audio_features_fn, split='val', batch_size=batch_size,
        shuffle=False))

if args.asr_model_dir:
    net = torch.load(os.path.join(args.asr_model_dir, 'net.best.pt'))
else:
    logging.info('Building ASR/SLT model')
    config = M1.get_default_config(hidden_size_factor=args.hidden_size_factor)
    net = M1.SpeechTranscriber(config)
    run_config = dict(max_norm=2.0, max_lr=args.cyclic_lr_max, min_lr=args.cyclic_lr_min, epochs=args.epochs,
                      l2_regularization=args.l2_regularization,
                      loss_logging_interval=args.loss_logging_interval,
                      validation_interval=args.validation_interval,
                      opt=args.optimizer
                      )
    logging.info('Training ASR/SLT')
    if data['train'].dataset.is_slt():
        M1.experiment(net, data, run_config, slt=True)
        copyfile('result.json', 'result_asr.json')
        copy_best('.', 'result_asr.json', 'asr.best.pt', experiment_type='slt')
    else:
        M1.experiment(net, data, run_config)
        copyfile('result.json', 'result_asr.json')
        copy_best('.', 'result_asr.json', 'asr.best.pt', experiment_type='asr')
    net = torch.load('asr.best.pt')

logging.info('Extracting ASR/SLT transcriptions')
hyp_asr, _ = extract_trn(net, data['val'].dataset, use_beam_decoding=True)

if args.text_image_model_dir:
    net = torch.load(os.path.join(args.text_image_model_dir,
                                  'net.best.pt'))
else:
    logging.info('Building model text-image')
    net = M2.TextImage(M2.get_default_config(hidden_size_factor=args.hidden_size_factor))
    run_config = dict(max_lr=args.cyclic_lr_max, min_lr=args.cyclic_lr_min, epochs=args.epochs,
                      l2_regularization=args.l2_regularization,
                      loss_logging_interval=args.loss_logging_interval,
                      validation_interval=args.validation_interval,
                      opt=args.optimizer
                      )
    logging.info('Training text-image')
    M2.experiment(net, data, run_config)
    copyfile('result.json', 'result_text_image.json')
    copy_best('.', 'result_text_image.json', 'ti.best.pt')
    net = torch.load('ti.best.pt')

logging.info('Evaluating text-image with ASR/SLT\'s output')
data = data['val'].dataset.evaluation()
correct = data['correct'].cpu().numpy()
image_e = net.embed_image(data['image'])
text_e = net.embed_text(hyp_asr)
result = E.ranking(image_e, text_e, correct)
res_out = dict(medr=np.median(result['ranks']),
               recall={1: np.mean(result['recall'][1]),
                       5: np.mean(result['recall'][5]),
                       10: np.mean(result['recall'][10])})
json.dump(res_out, open('result.json', 'w'))
