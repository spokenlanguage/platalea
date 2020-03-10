import logging
import pickle
from shutil import copyfile
import torch
import torch.nn as nn

import platalea.asr as M
import platalea.dataset as D
from utils.copy_best import copy_best

torch.manual_seed(123)


batch_size = 8
hidden_size = 1024
dropout = 0.0
feature_fname = 'mfcc_delta_features.pt'

logging.basicConfig(level=logging.INFO)

factors = [3, 9, 27, 81, 243]
lz = len(str(abs(factors[-1])))
for ds_factor in factors:
    logging.info('Loading data')
    data = dict(
        train=D.flickr8k_loader(split='train', batch_size=batch_size,
                                shuffle=True, feature_fname=feature_fname,
                                downsampling_factor=ds_factor),
        val=D.flickr8k_loader(split='val', batch_size=batch_size,
                              shuffle=False, feature_fname=feature_fname))
    fd = D.Flickr8KData
    fd.init_vocabulary(data['train'].dataset)

    # Saving config
    pickle.dump(dict(feature_fname=feature_fname,
                     label_encoder=fd.get_label_encoder(),
                     language='en'),
                open('config.pkl', 'wb'))

    config = dict(
        SpeechEncoder=dict(
            conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2,
                      padding=0, bias=False),
            rnn=dict(input_size=64, hidden_size=hidden_size, num_layers=6,
                     bidirectional=True, dropout=dropout),
            rnn_layer_type=nn.GRU),
        TextDecoder=dict(
            emb=dict(num_embeddings=fd.vocabulary_size(),
                     embedding_dim=hidden_size),
            drop=dict(p=dropout),
            att=dict(in_size_enc=hidden_size * 2, in_size_state=hidden_size,
                     hidden_size=hidden_size),
            rnn=dict(input_size=hidden_size * 3, hidden_size=hidden_size,
                     num_layers=1, dropout=dropout),
            out=dict(in_features=hidden_size * 3,
                     out_features=fd.vocabulary_size()),
            rnn_layer_type=nn.GRU,
            max_output_length=400,  # max length for flickr annotations is 199
            sos_id=fd.get_token_id(fd.sos),
            eos_id=fd.get_token_id(fd.eos),
            pad_id=fd.get_token_id(fd.pad)),
        inverse_transform_fn=fd.get_label_encoder().inverse_transform)

    logging.info('Building model')
    net = M.SpeechTranscriber(config)
    run_config = dict(max_norm=2.0, max_lr=2 * 1e-4, epochs=32, opt='adam')

    logging.info('Training')
    M.experiment(net, data, run_config)
    res_fname = 'result_{}.json'.format(ds_factor)
    copyfile('result.json', res_fname)
    copy_best(res_fname, 'net_{}.best.pt'.format(str(ds_factor).zfill(lz)),
              metric_accessor='asr')
