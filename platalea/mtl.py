from collections import Counter
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from platalea.basic import cyclic_scheduler
from platalea.encoders import SpeechEncoderBottom, SpeechEncoderSplit
from platalea.basic import SpeechImage
from platalea.asr import SpeechTranscriber
import platalea.loss
import platalea.score


class MTLNet(nn.Module):
    def __init__(self, config):
        super(MTLNet, self).__init__()
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
            TextDecoder=config['TextDecoder'],
            inverse_transform_fn=config['inverse_transform_fn']))
        self.lmbd = config.get('lmbd', 0.5)

    def cost(self, item):
        loss_si = self.SpeechImage.cost(item)
        loss_asr = self.SpeechTranscriber.cost(item)
        loss = self.lmbd * loss_si + (1 - self.lmbd) * loss_asr
        return loss, {'asr': loss_asr.item(), 'si': loss_si.item()}


def experiment_parallel(net, data, config):
    def val_loss():
        with torch.no_grad():
            net.eval()
            result = []
            for item in data['val']:
                item = {key: value.cuda() for key, value in item.items()}
                result.append(net.cost(item)[0].item())
            net.train()
        return torch.tensor(result).mean()

    net.cuda()
    net.train()
    # Preparing optimizer
    if 'lr' in config.keys():
        lr = config['lr']
    else:
        lr = 1.0
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = cyclic_scheduler(optimizer, len(data['train']),
                                 max_lr=config['max_lr'], min_lr=1e-6)
    optimizer.zero_grad()

    with open("result.json", "w") as out:
        for epoch in range(1, config['epochs']+1):
            cost = Counter()
            for j, item in enumerate(data['train'], start=1):
                item = {key: value.cuda() for key, value in item.items()}
                loss, loss_details = net.cost(item)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), config['max_norm'])
                optimizer.step()
                scheduler.step()
                cost += Counter({'cost': loss.item(), 'N': 1})
                cost += Counter(loss_details)
                if j % 100 == 0:
                    logging.info("train {} {} {} {}".format(
                        epoch, j, cost['cost'] / cost['N'],
                        [cost[k] / cost['N'] for k in loss_details.keys()]))
                if j % 400 == 0:
                    logging.info("valid {} {} {}".format(epoch, j, val_loss()))
            with torch.no_grad():
                net.eval()
                result = platalea.score.score(net.SpeechImage,
                                              data['val'].dataset)
                result.update(platalea.score.score_asr(net.SpeechTranscriber,
                                                       data['val'].dataset))
                net.train()
            result['epoch'] = epoch
            print(result, file=out, flush=True)
            logging.info("Saving model in net.{}.pt".format(epoch))
            torch.save(net, "net.{}.pt".format(epoch))


def val_loss(net, data):
    with torch.no_grad():
        net.eval()
        result = []
        for item in data['val']:
            item = {key: value.cuda() for key, value in item.items()}
            result.append(net.cost(item).item())
        net.train()
    return torch.tensor(result).mean()


def experiment(net, tasks, config):
    for t in tasks:
        # Preparing nets
        t['net'].cuda()
        t['net'].train()
        # Preparing optimizer
        if 'lr' in config.keys():
            lr = config['lr']
        else:
            lr = 1.0
        t['optimizer'] = optim.Adam(t['net'].parameters(), lr=lr)
        t['scheduler'] = cyclic_scheduler(t['optimizer'],
                                          len(t['data']['train']),
                                          max_lr=config['max_lr'], min_lr=1e-6)
        t['optimizer'].zero_grad()

    with open("result.json", "w") as out:
        for epoch in range(1, config['epochs']+1):
            for t in tasks:
                t['cost'] = Counter()
            for j, items in enumerate(zip(*[t['data']['train'] for t in tasks]),
                                      start=1):
                for i_t, t in enumerate(tasks):
                    item = {key: value.cuda() for key, value in items[i_t].items()}
                    loss = t['net'].cost(item)
                    t['optimizer'].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(t['net'].parameters(),
                                             config['max_norm'])
                    t['optimizer'].step()
                    t['scheduler'].step()
                    t['cost'] += Counter({'cost': loss.item(), 'N': 1})
                    if j % 100 == 0:
                        logging.info("train {} {} {} {}".format(
                            t['name'], epoch, j,
                            t['cost']['cost'] / t['cost']['N']))
                    if j % 400 == 0:
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
            result['epoch'] = epoch
            print(result, file=out, flush=True)
            # Saving model
            logging.info("Saving model in net.{}.pt".format(epoch))
            torch.save(net, "net.{}.pt".format(epoch))
