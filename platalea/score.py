import numpy as np
import platalea.dataset as D
import platalea.rank_eval as E
import platalea.xer as xer
import torch


def score(net, dataset):
    data = dataset.evaluation()
    correct = data['correct'].cpu().numpy()
    image_e = net.embed_image(data['image'])
    audio_e = net.embed_audio(data['audio'])
    result = E.ranking(image_e, audio_e, correct)
    return dict(medr=np.median(result['ranks']),
                recall={1: np.mean(result['recall'][1]),
                        5: np.mean(result['recall'][5]),
                       10: np.mean(result['recall'][10])})


def score_asr(net, dataset, beam_size=None):
    data = dataset.evaluation()
    trn = net.transcribe(data['audio'], beam_size=beam_size)
    ref = data['text']
    cer = xer.cer(trn, ref)
    wer = xer.wer(trn, ref)
    return dict(wer=wer, cer=cer)
