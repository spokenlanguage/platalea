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


def experiment(net, data, config):
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
    if 'lr' in config.keys():
        lr = config['lr']
    else:
        lr = 1.0
    if 'opt' in config.keys() and config['opt'] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = cyclic_scheduler(optimizer, len(data['train']),
                                     max_lr=config['max_lr'], min_lr=1e-6)
    else:
        optimizer = optim.Adadelta(net.parameters(), lr=lr, rho=0.95, eps=1e-8)
    optimizer.zero_grad()

    with open("result.json", "w") as out:
        best_wer = None
        for epoch in range(1, config['epochs']+1):
            cost = Counter()
            for j, item in enumerate(data['train'], start=1):
                item = {key: value.cuda() for key, value in item.items()}
                loss, loss_details = net.cost(item)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), config['max_norm'])
                optimizer.step()
                if 'opt' in config.keys() and config['opt'] == 'adam':
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
            if 'epsilon_decay' in config.keys():
                wer = result['wer']['WER']
                if best_wer is None or wer < best_wer:
                    best_wer = wer
                else:
                    net.load_state_dict(torch.load('net.{}.pt'.format(epoch - 1)))
                    for p in optimizer.param_groups:
                        p["eps"] *= config['epsilon_decay']
                        print('Epsilon decay - new value: ', p["eps"])
                logging.info("Saving model in net.{}.pt".format(epoch))
                # Saving weights only
                torch.save(net.state_dict(), "net.{}.pt".format(epoch))
            else:
                logging.info("Saving model in net.{}.pt".format(epoch))
                torch.save(net, "net.{}.pt".format(epoch))
    if 'epsilon_decay' in config.keys():
        # Save full model for inference
        torch.save(net, 'net.best.pt')
