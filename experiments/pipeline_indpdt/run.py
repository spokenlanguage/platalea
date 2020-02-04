import logging
import numpy as np
import pickle
from shutils import copyfile
import torch

import platalea.asr as M1
import platalea.dataset as D
import platalea.rank_eval as E
import platalea.text_image as M2
from platalea.util.extract_transcriptions import extract_trn

torch.manual_seed(123)


batch_size = 8
feature_fname = 'mfcc_delta_features.pt'

logging.basicConfig(level=logging.INFO)

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(split='train', batch_size=batch_size, shuffle=True,
                            feature_fname=feature_fname),
    val=D.flickr8k_loader(split='val', batch_size=batch_size, shuffle=False,
                          feature_fname=feature_fname))
fd = D.Flickr8KData
fd.init_vocabulary(data['train'].dataset)

# Saving config
pickle.dump(dict(feature_fname=feature_fname,
                 label_encoder=fd.get_label_encoder(),
                 language='en'),
            open('config.pkl', 'wb'))

logging.info('Building ASR model')
config = M1.get_defaul_config()
net = M1.SpeechTranscriber(config)
run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=32, opt='adam')

logging.info('Training ASR')
M1.experiment(net, data, run_config)
for i in range(32):
    copyfile('net.{}.pt'.format(i), 'asr.{}.pt'.format(i))

logging.info('Extracting ASR transcription')
hyp_asr, _ = extract_trn(net, data['val'], use_beam_decoding=True)

logging.info('Building model text-image')
net = M2.TextImage(M2.DEFAULT_CONFIG)
run_config = dict(max_lr=2 * 1e-4, epochs=32)

logging.info('Training text-image')
M2.experiment(net, data, run_config)
for i in range(32):
    copyfile('net.{}.pt'.format(i), 'text-image.{}.pt'.format(i))

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
