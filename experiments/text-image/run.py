import logging
import pickle
import torch

import platalea.text_image as M
import platalea.dataset as D

torch.manual_seed(123)


batch_size = 32
hidden_size = 1024
dropout = 0.0
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

logging.info('Building model')
net = M.TextImage(M.get_defaul_config())
run_config = dict(max_lr=2 * 1e-4, epochs=32)

logging.info('Training')
M.experiment(net, data, run_config)
