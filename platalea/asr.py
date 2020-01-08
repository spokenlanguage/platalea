from collections import Counter
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from platalea.basic import cyclic_scheduler
import platalea.dataset as D
from platalea.decoders import TextDecoder
from platalea.encoders import SpeechEncoder, SpeechEncoderVGG, SpeechEncoderMultiConv
import platalea.loss
import platalea.score


class SpeechTranscriber(nn.Module):
    def __init__(self, config):
        super(SpeechTranscriber, self).__init__()
        self.config = config
        if 'SpeechEncoder' in config:
            self.SpeechEncoder = SpeechEncoder(config['SpeechEncoder'])
        elif 'SpeechEncoderVGG' in config:
            self.SpeechEncoder = SpeechEncoderVGG(config['SpeechEncoderVGG'])
        elif 'SpeechEncoderMultiConv' in config:
            self.SpeechEncoder = SpeechEncoderMultiConv(config['SpeechEncoderMultiConv'])
        else:
            raise ValueError('Unknown encoder')
        self.TextDecoder = TextDecoder(config['TextDecoder'])
        self.inverse_transform_fn = config['inverse_transform_fn']

    def forward(self, speech, seq_len, target=None):
        out = self.SpeechEncoder(speech, seq_len)
        pred, attn_weights = self.TextDecoder.decode(out, target)
        return pred, attn_weights

    def transcribe(self, audio):
        audio = torch.utils.data.DataLoader(dataset=audio, batch_size=32,
                                            shuffle=False,
                                            collate_fn=D.batch_audio)
        trn = []
        for a, l in audio:
            pred, _ = self.forward(a.cuda(), l.cuda())
            trn.append(self.pred2trn(pred.detach().cpu()))
        trn = np.concatenate(trn)
        return trn

    def transcribe_beam(self, audio, audio_len, beam_size):
        audio = torch.utils.data.DataLoader(dataset=audio, batch_size=32,
                                            shuffle=False,
                                            collate_fn=D.batch_audio)
        trn = []
        for a, l in audio:
            enc_out = self.SpeechEncoder(a, l)
            preds = self.TextDecoder.beam_search(enc_out, beam_size)
            for i_seq in range(preds.shape[0]):
                seq = preds[i_seq]
                i_eos = (seq == self.TextDecoder.eos_id).nonzero()[0]
                i_last = i_eos[0] if i_eos.shape[0] > 0 else seq.shape[0]
                chars = [self.inverse_transform_fn(id.item()) for id in seq[:i_last]]
                trn.append(''.join(chars))
        trn = np.concatenate(trn)
        return trn

    def pred2trn(self, pred):
        trn = []
        ids = pred.argmax(dim=2)
        for i_seq in range(ids.shape[0]):
            seq = ids[i_seq]
            i_eos = (seq == self.TextDecoder.eos_id).nonzero()
            i_last = i_eos[0] if i_eos.shape[0] > 0 else seq.shape[0]
            chars = [self.inverse_transform_fn([id.item()])[0] for id in seq[:i_last]]
            trn.append(''.join(chars))
        return trn

    def cost(self, item):
        target = item['text'][:, 1:].contiguous()
        pred, _ = self.forward(item['audio'], item['audio_len'], target)

        # Masking padding
        # - flatten vectors
        target = target.view(-1)
        pred = pred.view(-1, self.TextDecoder.num_tokens)
        # - compute and apply mask
        mask = (target != self.TextDecoder.pad_id)
        target = target[mask]
        pred = pred[mask, :]

        cost = F.cross_entropy(pred, target)
        return cost


def experiment(net, data, config):
    def val_loss():
        with torch.no_grad():
            net.eval()
            result = []
            for item in data['val']:
                item = {key: value.cuda() for key, value in item.items()}
                result.append(net.cost(item).item())
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
    else:
        optimizer = optim.Adadelta(net.parameters(), lr=lr, rho=0.95, eps=1e-8)
    #scheduler = cyclic_scheduler(optimizer, len(data['train']), max_lr = config['max_lr'], min_lr = 1e-6)
    optimizer.zero_grad()

    with open("result.json", "w") as out:
        best_wer = None
        for epoch in range(1, config['epochs']+1):
            cost = Counter()
            for j, item in enumerate(data['train'], start=1):
                item = {key: value.cuda() for key, value in item.items()}
                loss = net.cost(item)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), config['max_norm'])
                optimizer.step()
                #scheduler.step()
                cost += Counter({'cost': loss.item(), 'N': 1})
                if j % 100 == 0:
                    logging.info("train {} {} {}".format(
                        epoch, j, cost['cost'] / cost['N']))
                if j % 400 == 0:
                    logging.info("valid {} {} {}".format(epoch, j, val_loss()))
            with torch.no_grad():
                net.eval()
                result = platalea.score.score_asr(net, data['val'].dataset)
                net.train()
            result['epoch'] = epoch
            print(result, file=out, flush=True)
            wer = result['wer']['WER']
            if best_wer is None or wer < best_wer:
                best_wer = wer
            else:
                net.load_state_dict(torch.load('net.{}.pt'.format(epoch - 1)))
                if 'epsilon_decay' in config.keys():
                    for p in optimizer.param_groups:
                        p["eps"] *= config['epsilon_decay']
                        print('Epsilon decay - new value: ', p["eps"])
            logging.info("Saving model in net.{}.pt".format(epoch))
            torch.save(net.state_dict(), "net.{}.pt".format(epoch))
    # Save full model for inference
    torch.save(net, 'net.best.pt')
