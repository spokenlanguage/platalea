import numpy as np
import torch
import torch.nn as nn
from platalea.encoders import SpeechEncoderVQ, SpeechEncoderVQ2, ImageEncoder, inout
import platalea.loss
from collections import Counter
import logging
import platalea.dataset as D
import platalea.score
import json

from platalea.optimizers import create_optimizer
from platalea.schedulers import create_scheduler


class SpeechImage(nn.Module):
    def __init__(self, config):
        super(SpeechImage, self).__init__()
        self.config = config
        # Components can be pre-instantiated or configured through a dictionary
        if config['SpeechEncoder'].get('VQEmbedding1', False):
            self.SpeechEncoder = SpeechEncoderVQ2(config['SpeechEncoder'])
        else:
            self.SpeechEncoder = SpeechEncoderVQ(config['SpeechEncoder'])
        self.ImageEncoder = ImageEncoder(config['ImageEncoder'])

    def cost(self, item):
        speech_enc = self.SpeechEncoder(item['audio'], item['audio_len'])
        image_enc = self.ImageEncoder(item['image'])
        scores = platalea.loss.cosine_matrix(speech_enc, image_enc)
        loss = platalea.loss.contrastive(scores, margin=self.config['margin_size'])
        return loss

    def embed_image(self, images):
        image = torch.utils.data.DataLoader(dataset=images, batch_size=32,
                                            shuffle=False,
                                            collate_fn=D.batch_image)
        image_e = []
        for i in image:
            image_e.append(self.ImageEncoder(i.cuda()).detach().cpu().numpy())
        image_e = np.concatenate(image_e)
        return image_e

    def embed_audio(self, audios):
        audio = torch.utils.data.DataLoader(dataset=audios, batch_size=32,
                                            shuffle=False,
                                            collate_fn=D.batch_audio)
        audio_e = []
        for a, l in audio:
            audio_e.append(self.SpeechEncoder(a.cuda(), l.cuda()).detach().cpu().numpy())
        audio_e = np.concatenate(audio_e)
        return audio_e

    def code_audio(self, audios, one_hot=False):  # FIXME messed up sized ETC
        audio = torch.utils.data.DataLoader(dataset=audios, batch_size=32,
                                            shuffle=False,
                                            collate_fn=D.batch_audio)
        audio_e = []
        for a, l in audio:
            if one_hot:
                codes = self.SpeechEncoder.Codebook(self.SpeechEncoder.Bottom(a.cuda(), l.cuda()))['one_hot']
            else:
                codes = self.SpeechEncoder.Codebook(self.SpeechEncoder.Bottom(a.cuda(), l.cuda()))['codes']
            codes = codes.detach().cpu().numpy()
            for code, L in zip(list(codes), list(l)):
                code = code[:inout(self.SpeechEncoder.Bottom.Conv, L).item()]
                audio_e.append(code)
        return audio_e


def experiment(net, data, config):
    def val_loss():
        net.eval()
        result = []
        for item in data['val']:
            item = {key: value.cuda() for key, value in item.items()}
            result.append(net.cost(item).item())
        net.train()
        return torch.tensor(result).mean()

    net.cuda()
    net.train()
    optimizer = create_optimizer(config, net.parameters())
    config['min_lr'] = 1e-6
    scheduler = create_scheduler(config, optimizer, data)

    results = []
    with open("result.json", "w") as out:
        for epoch in range(1, config['epochs']+1):
            cost = Counter()
            for j, item in enumerate(data['train'], start=1):  # check reshuffling
                item = {key: value.cuda() for key, value in item.items()}
                loss = net.cost(item)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                cost += Counter({'cost': loss.item(), 'N': 1})
                average_loss = cost['cost'] / cost['N']
                if j % config['loss_logging_interval'] == 0:
                    logging.info("train {} {} {}".format(epoch, j, average_loss))
                if j % config['validation_interval'] == 0:
                    logging.info("valid {} {} {}".format(epoch, j, val_loss()))
            result = platalea.score.score(net, data['val'].dataset)
            result['average_loss'] = average_loss
            result['epoch'] = epoch
            results.append(result)
            print(json.dumps(result), file=out, flush=True)
            logging.info("Saving model in net.{}.pt".format(epoch))
            torch.save(net, "net.{}.pt".format(epoch))
    return results


DEFAULT_CONFIG = dict(SpeechEncoder=dict(SpeechEncoderBottom=dict(conv=dict(in_channels=39, out_channels=64, kernel_size=6,
                                                                            stride=2, padding=0, bias=False),
                                                                  rnn=dict(input_size=64, hidden_size=1024, num_layers=2,
                                                                           bidirectional=True, dropout=0)),
                                         VQEmbedding=dict(num_codebook_embeddings=256, embedding_dim=1024, jitter=0.12),
                                         SpeechEncoderTop=dict(rnn=dict(input_size=64, hidden_size=1024, num_layers=2,
                                                                        bidirectional=True, dropout=0),
                                                               att=dict(in_size=2048, hidden_size=128))),
                      ImageEncoder=dict(linear=dict(in_size=2048, out_size=2*1024), norm=True),
                      margin_size=0.2)
