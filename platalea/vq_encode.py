import glob
from platalea.utils.preprocessing import audio_features
from zerospeech2020.evaluation import evaluation_2019
import logging
import os
import os.path
import numpy as np
import torch

config = dict(type='mfcc', delta=True, alpha=0.97, n_filters=40,  window_size=0.025, frame_shift=0.010)


def encode(net, datadir, outdir):
    paths = glob.glob(datadir + "/*.wav")
    assert len(paths) > 0
    try:
        feat = torch.load(datadir + "_audiofeat.pt")
    except FileNotFoundError:
        logging.info("Preprocessing data")
        feat = audio_features(paths, config)
        logging.info("Saving preprocessed data")
        torch.save(feat, str(datadir) + "_audiofeat.pt")
    logging.info("Computing codes")
    codes = net.code_audio(feat, one_hot=True)
    for path, code in zip(paths, codes):
        filename = os.path.splitext(os.path.basename(path))[0]
        out = outdir + '/' + filename + ".txt"
        assert code.shape[0] > 0
        np.savetxt(out, code.astype(int), fmt='%d')


def encode_zerospeech(net, outdir='.'):
    encode(net, "/roaming/gchrupal/verdigris/platalea.vq/data/2020/2019/english/test/")


def evaluate_zerospeech(net, outdir='.'):
    encode_zerospeech(net, outdir=outdir)
    logging.info("Data encoded")
    logging.info("Evaluating")
    scores = evaluation_2019.evaluate("{}/../../../".format(outdir),
                                      '/roaming/gchrupal/verdigris/platalea.vq/data/2020',
                                      ['english'],
                                      'cosine',
                                      True)
    return scores
