import torch
import torch.nn as nn
import logging
import platalea.asr as M
import platalea.dataset as D

torch.manual_seed(123)

batch_size = 8
hidden_size = 320
dropout = 0.0
feature_fname='mfcc_feature.pt'

logging.basicConfig(level=logging.INFO)

logging.info('Loading data')
data = dict(
    train=D.flickr8k_loader(split='train', batch_size=batch_size,
                            shuffle=True, feature_fname=feature_fname),
    val=D.flickr8k_loader(split='val', batch_size=batch_size, shuffle=False,
                          feature_fname=feature_fname))
fd = D.Flickr8KData
fd.init_vocabulary(data['train'].dataset)

config = dict(
    SpeechEncoder=dict(
        conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2,
                  padding=6, bias=False),
        rnn=dict(input_size=64, hidden_size=hidden_size, num_layers=2,
                 bidirectional=True, dropout=dropout),
        rnn_layer_type=nn.LSTM),
    TextDecoder=dict(
        emb=dict(num_embeddings=fd.vocabulary_size(), embedding_dim=hidden_size),
        drop=dict(p=dropout),
        att=dict(in_size_enc=hidden_size * 2, in_size_state=hidden_size,
                 hidden_size=hidden_size),
        rnn=dict(input_size=hidden_size * 3, hidden_size=hidden_size,
                 num_layers=1, dropout=dropout),
        out=dict(in_features=hidden_size * 3,
                 out_features=fd.vocabulary_size()),
        rnn_layer_type=nn.LSTM,
        max_output_length=400,  # max length for flickr annotations is 199
        sos_id=fd.sos,
        eos_id=fd.eos,
        pad_id=fd.pad),
    inverse_transform_fn=fd.le.inverse_transform)

logging.info('Building model')
net = M.SpeechTranscriber(config)
run_config = dict(max_norm=2.0, epochs=32, epsilon_decay=0.01)

logging.info('Training')
M.experiment(net, data, run_config)
