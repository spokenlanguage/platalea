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
                 label_encoder=fd.get_label_encoder()),
            open('config.pkl', 'wb'))

config = dict(
    TextEncoder=dict(
        emb=dict(num_embeddings=fd.vocabulary_size(),
                 embedding_dim=128),
        rnn=dict(input_size=128, hidden_size=hidden_size, num_layers=2,
                 bidirectional=True, dropout=0),
        att=dict(in_size=hidden_size * 2, hidden_size=128)),
    ImageEncoder=dict(
        linear=dict(in_size=hidden_size * 2, out_size=hidden_size * 2),
        norm=True),
    margin_size=0.2)

logging.info('Building model')
net = M.TextImage(config)
run_config = dict(max_lr=2 * 1e-4, epochs=32)

logging.info('Training')
M.experiment(net, data, run_config)
