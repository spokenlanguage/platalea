import numpy as np
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


def score_text_image(net, dataset):
    data = dataset.evaluation()
    correct = data['correct'].cpu().numpy()
    image_e = net.embed_image(data['image'])
    text_e = net.embed_text(data['text'])
    result = E.ranking(image_e, text_e, correct)
    return dict(medr=np.median(result['ranks']),
                recall={1: np.mean(result['recall'][1]),
                        5: np.mean(result['recall'][5]),
                        10: np.mean(result['recall'][10])})


def score_speech_text(net, dataset):
    data = dataset.evaluation()
    audio_e = net.embed_audio(data['audio'])
    text_e = net.embed_text(data['text'])
    correct = torch.eye(len(data['audio'])).type(torch.bool)
    result = E.ranking(audio_e, text_e, correct)
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


def bleu_score(references, hypotheses):
    from nltk.translate.bleu_score import sentence_bleu
    bleu = np.zeros(len(references))
    for i in range(len(references)):
        bleu[i] = sentence_bleu([references[i]], hypotheses[i])
    return bleu.mean()


def score_slt(net, dataset, beam_size=None):
    data = dataset.evaluation()
    trn = net.transcribe(data['audio'], beam_size=beam_size)
    ref = data['text']
    cer = xer.cer(trn, ref)
    trn = dataset.split_sentences(trn)
    ref = dataset.split_sentences(ref)
    bleu = bleu_score(ref, trn)
    return dict(bleu=bleu, cer=cer)
