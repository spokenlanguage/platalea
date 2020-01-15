import numpy as np
import platalea.dataset as D
import platalea.rank_eval as E
import platalea.xer as xer
import torch


# TODO: delete after implementation of score_mtl
def batch_process(dataloader, fn):
    res = [fn(*d) if type(d) == tuple else fn(d) for d in dataloader]
    if type(res[0]) == tuple:
        res = list(zip(*res))
        return tuple([np.concatenate(r) for r in res])
    else:
        return np.concatenate(res)


def score(net, dataset):
    data = dataset.evaluation()
    # Extract embeddings for images and audio
    image_dl = torch.utils.data.DataLoader(dataset=data['image'],
                                           batch_size=32, shuffle=False,
                                           collate_fn=D.batch_image)
    image_e = np.concatenate([net.embed_image(d) for d in image_dl])
    audio_dl = torch.utils.data.DataLoader(dataset=data['audio'],
                                           batch_size=32, shuffle=False,
                                           collate_fn=D.batch_audio)
    audio_e = np.concatenate([net.embed_audio(*d) for d in audio_dl])
    # Evaluate
    correct = data['correct'].cpu().numpy()
    result = E.ranking(image_e, audio_e, correct)
    return dict(medr=np.median(result['ranks']),
                recall={1: np.mean(result['recall'][1]),
                        5: np.mean(result['recall'][5]),
                       10: np.mean(result['recall'][10])})


def score_asr(net, dataset, use_beam=False):
    data = dataset.evaluation()
    audio_dl = torch.utils.data.DataLoader(dataset=data['audio'],
                                           batch_size=32, shuffle=False,
                                           collate_fn=D.batch_audio)
    if use_beam:
        fn = lambda x, y: net.transcribe_beam(x, y, beam_size=10)
    else:
        fn = net.transcribe
    trn = np.concatenate([fn(*d) for d in audio_dl])
    ref = [txt['raw'] for txt in data['text']]
    cer = xer.cer(trn, ref)
    wer = xer.wer(trn, ref)
    return dict(wer=wer, cer=cer)


#def score_mtl(net, dataset, use_beam=False):
#    data = dataset.evaluation()
#    if use_beam:
#        trn = net.transcribe_beam(data['audio'])
#    else:
#        trn = net.transcribe(data['audio'])
#    ref = [txt['raw'] for txt in data['text']]
#    cer = xer.cer(trn, ref)
#    wer = xer.wer(trn, ref)
#    return dict(wer=wer, cer=cer)
#
#    # Extract encodings and transcriptions
#    audio = torch.utils.data.DataLoader(dataset=data['audio'], batch_size=32,
#                                        shuffle=False,
#                                        collate_fn=D.batch_audio)
#    image = torch.utils.data.DataLoader(dataset=data['image'], batch_size=32,
#                                        shuffle=False,
#                                        collate_fn=D.batch_image)
#    speech_enc = []
#    image_enc = []
#    trn = []
#    i = 0
#    for (a, l), i in zip(audio, image):
#        if i == 4:
#            break
#        # FIXME: forward only works for 1 step
#        se, ie, tp = net.forward(a, l, i)
#        speech_enc.append(se.detach().cpu().numpy())
#        image_enc.append(ie.detach().cpu().numpy()))
#        trn.append(self.pred2trn(text_pred.detach().cpu()))
#        i += 1
#    audio_enc = np.concatenate(audio_enc)
#    image_enc = np.concatenate(image_enc)
#    trn = np.concatenate(trn)
#
#    # Compute metrics
#    result = E.ranking(image_e, audio_e, correct)
#    ref = [txt['raw'] for txt in data['text']]
#    cer = xer.cer(trn, ref)
#    wer = xer.wer(trn, ref)
#    return dict(medr=np.median(result['ranks']),
#                recall={1: np.mean(result['recall'][1]),
#                        5: np.mean(result['recall'][5]),
#                       10: np.mean(result['recall'][10])},
#                wer=wer,
#                cer=cer)
