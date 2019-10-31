import torch
import platalea.rank_eval as E
import numpy as np
import logging

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


