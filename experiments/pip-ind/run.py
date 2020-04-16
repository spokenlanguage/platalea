import argparse
import logging
import numpy as np
import pickle
import os
from shutil import copyfile
import torch

import platalea.asr as M1
import platalea.dataset as D
import platalea.rank_eval as E
import platalea.text_image as M2
from utils.copy_best import copy_best
from utils.extract_transcriptions import extract_trn

torch.manual_seed(123)


batch_size = 8
feature_fname = 'mfcc_delta_features.pt'

logging.basicConfig(level=logging.INFO)

# Parse command line parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    '--slt_model_dir',
    help='Path to the directory where the pretrained SLT model is stored',
    dest='slt_model_dir', type=str, action='store')
parser.add_argument(
    '--text_image_model_dir',
    help='Path to the directory where the pretrained text-image model is \
    stored',
    dest='text_image_model_dir', type=str, action='store')
args = parser.parse_args()

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(split='train', batch_size=batch_size, shuffle=True,
                            feature_fname=feature_fname),
    val=D.flickr8k_loader(split='val', batch_size=batch_size, shuffle=False,
                          feature_fname=feature_fname))
fd = D.Flickr8KData
if args.slt_model_dir:
    config_fpath = os.path.join(args.slt_model_dir, 'config.pkl')
    config = pickle.load(open(config_fpath, 'rb'))
    fd.le = config['label_encoder']
else:
    fd.init_vocabulary(data['train'].dataset)
    # Saving config
    pickle.dump(dict(feature_fname=feature_fname,
                     label_encoder=fd.get_label_encoder(),
                     language='en'),
                open('config.pkl', 'wb'))

if args.slt_model_dir:
    net = torch.load(os.path.join(args.slt_model_dir, 'net.best.pt'))
else:
    logging.info('Building SLT model')
    config = M1.get_default_config()
    net = M1.SpeechTranscriber(config)
    run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=32, opt='adam')
    logging.info('Training SLT')
    M1.experiment(net, data, run_config, slt=True)
    copyfile('result.json', 'result_slt.json')
    copy_best('result_slt.json', 'slt.best.pt', experiment_type='slt')
    net = torch.load('slt.best.pt')

logging.info('Extracting SLT transcriptions')
hyp_slt, _ = extract_trn(net, data['val'].dataset, use_beam_decoding=True)

if args.text_image_model_dir:
    config_fpath = os.path.join(args.text_image_model_dir, 'config.pkl')
    config = pickle.load(open(config_fpath, 'rb'))
    fd.le = config['label_encoder']
elif args.slt_model_dir:
    # Saving config for text-image model
    pickle.dump(dict(feature_fname=feature_fname,
                     label_encoder=fd.get_label_encoder(),
                     language='en'),
                open('config.pkl', 'wb'))

if args.text_image_model_dir:
    net = torch.load(os.path.join(args.text_image_model_dir, 'net.best.pt'))
else:
    logging.info('Building model text-image')
    net = M2.TextImage(M2.get_default_config())
    run_config = dict(max_lr=2 * 1e-4, epochs=32)
    logging.info('Training text-image')
    M2.experiment(net, data, run_config)
    copyfile('result.json', 'result_text_image.json')
    copy_best('result_text_image.json', 'ti.best.pt')
    net = torch.load('ti.best.pt')

logging.info('Evaluating text-image with SLT\'s output')
data = data['val'].dataset.evaluation()
correct = data['correct'].cpu().numpy()
image_e = net.embed_image(data['image'])
text_e = net.embed_text(hyp_slt)
result = E.ranking(image_e, text_e, correct)
print(dict(medr=np.median(result['ranks']),
           recall={1: np.mean(result['recall'][1]),
                   5: np.mean(result['recall'][5]),
                   10: np.mean(result['recall'][10])}))
