import argparse
import json
import logging
import torch

import platalea.dataset as D
import platalea.score

torch.manual_seed(123)

batch_size = 16
feature_fname = 'mfcc_delta_features.pt'
limit = None

logging.basicConfig(level=logging.INFO)

# Parse command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('path', metavar='path', help='Model\'s path', nargs='+')
parser.add_argument('-b', help='Use beam decoding', dest='use_beam_decoding',
                    action='store_true', default=False)
parser.add_argument('-t', help='Test mode', dest='testmode',
                    action='store_true', default=False)
args = parser.parse_args()

# Setup test mode
if args.testmode:
    limit = 100

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(split='train', batch_size=batch_size,
                            shuffle=True, feature_fname=feature_fname),
    val=D.flickr8k_loader(split='val', batch_size=batch_size, shuffle=False,
                          feature_fname=feature_fname))
fd = D.Flickr8KData
fd.init_vocabulary(data['train'].dataset)

for path in args.path:
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
