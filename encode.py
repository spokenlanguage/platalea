import glob
import torch
from platalea.preprocessing import audio_features
model = torch.load("experiments/vq-256-q1/net.32.pt")

import torch
torch.manual_seed(123)
import logging
import platalea.basicvq as M
import platalea.dataset as D

logging.basicConfig(level=logging.INFO)

logging.info('Loading data')
data = dict(train=D.flickr8k_loader(split='train', batch_size=32, shuffle=True),
            val=D.flickr8k_loader(split='val', batch_size=32, shuffle=False))
D.Flickr8KData.init_vocabulary(data['train'].dataset)

bidi = True
config = dict(SpeechEncoder=dict(SpeechEncoderBottom=dict(conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2, padding=0, bias=False),
                                                          rnn= dict(input_size=64, hidden_size=1024, num_layers=1,
                                                                    bidirectional=bidi, dropout=0)),
                                 VQEmbedding=dict(num_codebook_embeddings=256, embedding_dim=2 * 1024 if bidi else 1024, jitter=0.12),
                                 SpeechEncoderTop=dict(rnn= dict(input_size=2 * 1024 if bidi else 1024, hidden_size=1024, num_layers=3,
                                                                 bidirectional=bidi, dropout=0),
                                                       att= dict(in_size=2048, hidden_size=128))),
              ImageEncoder=dict(linear=dict(in_size=2048, out_size=2*1024), norm=True),
              margin_size=0.2)


logging.info('Building model')
net = M.SpeechImage(config)
net.load_state_dict(model.state_dict())
net.cuda()


import os.path
import numpy as np

paths = glob.glob("/roaming/gchrupal/verdigris/zerospeech2020/2020/2019/english/test/*.wav")
config = dict(type='mfcc', delta=True, alpha=0.97, n_filters=40,  window_size=0.025, frame_shift=0.010)
res = audio_features(paths, config)


codes = net.code_audio(res, one_hot=True)

for path, code in zip(paths, codes):
    filename = os.path.splitext(os.path.basename(path))[0]
    out = "fufi/" + filename + ".txt"
    logging.info("Writing {}".format(out))
    np.savetxt(out, code.astype(int), fmt='%d')




