import logging
import os
import random
from shutil import copyfile
import torch

import platalea.asr as M1
import platalea.dataset as D
import platalea.text_image as M2
from platalea.utils.copy_best import copy_best
from platalea.utils.extract_transcriptions import extract_trn
from platalea.experiments.config import args


# Parsing arguments
args.add_argument(
    '--asr_model_dir',
    help='Path to the directory where the pretrained ASR/SLT model is stored',
    dest='asr_model_dir', type=str, action='store')
args.enable_help()
args.parse()

# Setting general configuration
torch.manual_seed(args.seed)
random.seed(args.seed)
logging.basicConfig(level=logging.INFO)


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
        shuffle=False, downsampling_factor=args.downsampling_factor))

if args.asr_model_dir:
    net = torch.load(os.path.join(args.asr_model_dir, 'net.best.pt'))
else:
    logging.info('Building ASR/SLT model')
    config = M1.get_default_config()
    net = M1.SpeechTranscriber(config)
    run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=32)
    logging.info('Training ASR/SLT')
    if data['train'].dataset.is_slt():
        M1.experiment(net, data, run_config, slt=True)
        copy_best('result.json', 'asr.best.pt', experiment_type='slt')
    else:
        M1.experiment(net, data, run_config)
        copy_best('result.json', 'asr.best.pt', experiment_type='asr')
    copyfile('result.json', 'result_asr.json')
    net = torch.load('asr.best.pt')

logging.info('Extracting ASR/SLT transcriptions')
for set_name in ['train', 'val']:
    ds = data[set_name].dataset
    hyp_asr, ref_asr = extract_trn(net, ds, use_beam_decoding=True)
    # Replacing original transcriptions with ASR/SLT's output
    for i in range(len(hyp_asr)):
        item = ds.split_data[i]
        if item[2] == ref_asr[i]:
            ds.split_data[i] = (item[0], item[1], hyp_asr[i])
        else:
            msg = 'Extracted reference #{} ({}) doesn\'t match dataset\'s \
                    one ({}) for {} set.'
            msg = msg.format(i, ref_asr[i], ds.split_data[i][3], set_name)
            logging.warning(msg)

logging.info('Building model text-image')
net = M2.TextImage(M2.get_default_config())
run_config = dict(max_lr=2 * 1e-4, epochs=32)

logging.info('Training text-image')
M2.experiment(net, data, run_config)
copyfile('result.json', 'result_text_image.json')
copy_best('result_text_image.json', 'ti.best.pt')
