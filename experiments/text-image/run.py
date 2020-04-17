import logging
import pickle
import torch

import platalea.text_image as M
import platalea.dataset as D

torch.manual_seed(123)


batch_size = 32
hidden_size = 1024
dropout = 0.0

logging.basicConfig(level=logging.INFO)

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(split='train', batch_size=batch_size, shuffle=True),
    val=D.flickr8k_loader(split='val', batch_size=batch_size, shuffle=False))
fd = D.Flickr8KData
fd.init_vocabulary(data['train'].dataset)

# Saving config
pickle.dump(data['train'].dataset.get_config(),
            open('config.pkl', 'wb'))

logging.info('Building model')
net = M.TextImage(M.get_default_config())
run_config = dict(max_lr=2 * 1e-4, epochs=32)

logging.info('Training')
M.experiment(net, data, run_config)
