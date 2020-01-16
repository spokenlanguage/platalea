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


def score_asr(net, dataset, use_beam=False):
    data = dataset.evaluation()
    if use_beam:
        trn = net.transcribe_beam(data['audio'])
    else:
        trn = net.transcribe(data['audio'])
    ref = [txt['raw'] for txt in data['text']]
    cer = xer.cer(trn, ref)
    wer = xer.wer(trn, ref)
    return dict(wer=wer, cer=cer)
