from collections import Counter
import json
import logging
import torch
import torch.nn as nn

import platalea.schedulers
from platalea.encoders import SpeechEncoderBottom, SpeechEncoderSplit
from platalea.basic import SpeechImage
from platalea.speech_text import SpeechText
from platalea.asr import SpeechTranscriber
import platalea.loss
import platalea.score
import platalea.hardware
from platalea.optimizers import create_optimizer
from platalea.schedulers import create_scheduler


class MTLNetASR(nn.Module):
    def __init__(self, config):
        super(MTLNetASR, self).__init__()
        self.config = config
        SharedEncoder = SpeechEncoderBottom(config['SharedEncoder'])
        SpeechEncoderSplitSI = SpeechEncoderSplit(dict(
            SpeechEncoderBottom=SharedEncoder,
            SpeechEncoderTop=config['SpeechEncoderTopSI']))
        SpeechEncoderSplitASR = SpeechEncoderSplit(dict(
            SpeechEncoderBottom=SharedEncoder,
            SpeechEncoderTop=config['SpeechEncoderTopASR']))
        self.SpeechImage = SpeechImage(dict(
            SpeechEncoder=SpeechEncoderSplitSI,
            ImageEncoder=config['ImageEncoder'],
            margin_size=config['margin_size']))
        self.SpeechTranscriber = SpeechTranscriber(dict(
            SpeechEncoder=SpeechEncoderSplitASR,
            TextDecoder=config['TextDecoder']))
        self.lmbd = config.get('lmbd', 0.5)

    def cost(self, item):
        loss_si = self.SpeechImage.cost(item)
        loss_asr = self.SpeechTranscriber.cost(item)
        loss = self.lmbd * loss_si + (1 - self.lmbd) * loss_asr
        return loss, {'asr': loss_asr.item(), 'speech-image': loss_si.item()}


class MTLNetSpeechText(nn.Module):
    def __init__(self, config):
        super(MTLNetSpeechText, self).__init__()
        self.config = config
        SharedEncoder = SpeechEncoderBottom(config['SharedEncoder'])
        SpeechEncoderSplitSI = SpeechEncoderSplit(dict(
            SpeechEncoderBottom=SharedEncoder,
            SpeechEncoderTop=config['SpeechEncoderTopSI']))
        SpeechEncoderSplitST = SpeechEncoderSplit(dict(
            SpeechEncoderBottom=SharedEncoder,
            SpeechEncoderTop=config['SpeechEncoderTopST']))
        self.SpeechImage = SpeechImage(dict(
            SpeechEncoder=SpeechEncoderSplitSI,
            ImageEncoder=config['ImageEncoder'],
            margin_size=config['margin_size']))
        self.SpeechText = SpeechText(dict(
            SpeechEncoder=SpeechEncoderSplitST,
            TextEncoder=config['TextEncoder'],
            margin_size=config['margin_size']))
        self.lmbd = config.get('lmbd', 0.5)

    def cost(self, item):
        loss_si = self.SpeechImage.cost(item)
        loss_st = self.SpeechText.cost(item)
        loss = self.lmbd * loss_si + (1 - self.lmbd) * loss_st
        return loss, {'speech-text': loss_st.item(),
                      'speech-image': loss_si.item()}


def val_loss(net, data):
    _device = platalea.hardware.device()
    with torch.no_grad():
        net.eval()
        result = []
        for item in data['val']:
            item = {key: value.to(_device) for key, value in item.items()}
            result.append(net.cost(item).item())
        net.train()
    return torch.tensor(result).mean()


def task_iterator(tasks):
    # returns a list of batches for each task to train this step
    # allows to train a task only every n step
    iterators = [t['data']['train'].__iter__() for t in tasks]
    step = 1
    try:
        while True:
            yield [(t, next(it)) for t, it in zip(tasks, iterators) if 'step' not in t or step % t['step'] == 0]
            step += 1
    except StopIteration:
        return


def experiment(net, tasks, config):
    _device = platalea.hardware.device()
    for t in tasks:
        # Preparing nets
        t['net'].to(_device)
        t['net'].train()
        t['optimizer'] = create_optimizer(config, t['net'].parameters())
        t['scheduler'] = create_scheduler(config, t['optimizer'], t['data'])

    results = []
    with open("result.json", "w") as out:
        for epoch in range(1, config['epochs']+1):
            for t in tasks:
                t['cost'] = Counter()
            for j, items in enumerate(task_iterator(tasks), start=1):
                for t, item in items:
                    item = {k: v.to(_device) for k, v in item.items()}
                    loss = t['net'].cost(item)
                    t['optimizer'].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(t['net'].parameters(),
                                             config['max_norm'])
                    t['optimizer'].step()
                    t['scheduler'].step()
                    t['cost'] += Counter({'cost': loss.item(), 'N': 1})
                    t['average_loss'] = t['cost']['cost'] / t['cost']['N']
                    if j % config['loss_logging_interval'] == 0:
                        logging.info("train {} {} {} {}".format(
                            t['name'], epoch, j, t['average_loss']))
                    if j % config['validation_interval'] == 0:
                        logging.info("valid {} {} {} {}".format(
                            t['name'], epoch, j,
                            val_loss(t['net'], t['data'])))
            # Evaluation
            result = {}
            with torch.no_grad():
                net.eval()
                for t in tasks:
                    result[t['name']] = t['eval'](t['net'],
                                                  t['data']['val'].dataset)
                net.train()
            for t in tasks:
                result[t['name']].update({'average_loss': t['average_loss']})
            result['epoch'] = epoch
            results.append(result)
            json.dump(result, out)
            print('', file=out, flush=True)
            # Saving model
            logging.info("Saving model in net.{}.pt".format(epoch))
            torch.save(net, "net.{}.pt".format(epoch))

    return results
