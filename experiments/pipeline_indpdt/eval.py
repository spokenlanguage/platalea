import logging
import numpy as np
import pickle
import torch

import platalea.asr as M1
import platalea.dataset as D
import platalea.rank_eval as E
import platalea.text_image as M2
from utils.extract_transcriptions import extract_trn

torch.manual_seed(123)


batch_size = 8

logging.basicConfig(level=logging.INFO)

# Loading ASR config
conf = pickle.load(open('config_asr.pkl', 'rb'))

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(split='train', batch_size=batch_size, shuffle=True,
                            feature_fname=conf['feature_fname']),
    val=D.flickr8k_loader(split='val', batch_size=batch_size, shuffle=False,
                          feature_fname=conf['feature_fname']))
fd = D.Flickr8KData
fd.le = conf['label_encoder']

logging.info('Loading ASR model')
net = torch.load('asr.best.pt')

logging.info('Extracting ASR transcription')
hyp_asr, _ = extract_trn(net, data['val'].dataset, use_beam_decoding=True)

# Loading ASR config
conf = pickle.load(open('config_text-image.pkl', 'rb'))
fd.le = conf['label_encoder']

logging.info('Loading text-image model')
net = torch.load('text-image.best.pt')

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
