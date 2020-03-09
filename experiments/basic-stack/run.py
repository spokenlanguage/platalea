import torch
torch.manual_seed(123)
import logging
import platalea.basic as M
import platalea.dataset as D

import configargparse as argparse

parser = argparse.ArgumentParser(description='Run the basic-stack experiment.')
parser.add_argument('--epochs', action='store', default=32, dest='epochs', type=int,
                   help='number of epochs after which to stop training (default: 32)')

args = parser.parse_args()


logging.basicConfig(level=logging.INFO)

logging.info('Loading data')
data = dict(train=D.flickr8k_loader(split='train', batch_size=32, shuffle=True),
            val=D.flickr8k_loader(split='val', batch_size=32, shuffle=False))
D.Flickr8KData.init_vocabulary(data['train'].dataset)

config = dict(SpeechEncoder=dict(conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2, padding=0, bias=False),
                                 rnn= dict(input_size=64, hidden_size=1024, num_layers=4, bidirectional=True, dropout=0), 
                                 att= dict(in_size=2048, hidden_size=128)),
              ImageEncoder=dict(linear=dict(in_size=2048, out_size=2*1024), norm=True),
              margin_size=0.2)

logging.info('Building model')
net = M.SpeechImage(config)
run_config = dict(max_lr=2 * 1e-4, epochs=args.epochs)

logging.info('Training')
M.experiment(net, data, run_config)
