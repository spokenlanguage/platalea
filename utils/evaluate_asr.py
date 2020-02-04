import argparse
import json
import logging
import pickle
import torch

import platalea.dataset as D
import platalea.score

torch.manual_seed(123)

batch_size = 16

logging.basicConfig(level=logging.INFO)

# Parse command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('path', metavar='path', help='Model\'s path')
parser.add_argument('-b', help='Use beam decoding', dest='use_beam_decoding',
                    action='store_true', default=False)
args = parser.parse_args()


# Loading config
conf = pickle.load(open('config.pkl', 'rb'))

logging.info('Loading data')
data = dict(val=D.flickr8k_loader(split='val', batch_size=batch_size,
                                  shuffle=False,
                                  feature_fname=conf['feature_fname']))
fd = D.Flickr8KData
fd.le = conf['label_encoder']

logging.info('Loading model')
net = torch.load(path)
logging.info('Evaluating')
with torch.no_grad():
    net.eval()
    if args.use_beam_decoding:
        result = platalea.score.score_asr(net, data['val'].dataset,
                                          beam_size=10)
    else:
        result = platalea.score.score_asr(net, data['val'].dataset)
print(json.dumps(result))
