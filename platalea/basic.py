from collections import Counter
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from platalea.encoders import SpeechEncoder, ImageEncoder
import platalea.loss
import platalea.dataset as D
import platalea.score
import platalea.hardware


_device = platalea.hardware.device()


class SpeechImage(nn.Module):
    def __init__(self, config):
        super(SpeechImage, self).__init__()
        self.config = config
        # Components can be pre-instantiated or configured through a dictionary
        if isinstance(config['SpeechEncoder'], nn.Module):
            self.SpeechEncoder = config['SpeechEncoder']
        else:
            self.SpeechEncoder = SpeechEncoder(config['SpeechEncoder'])
        if isinstance(config['ImageEncoder'], nn.Module):
            self.ImageEncoder = config['ImageEncoder']
        else:
            self.ImageEncoder = ImageEncoder(config['ImageEncoder'])

    def cost(self, item):
        speech_enc = self.SpeechEncoder(item['audio'], item['audio_len'])
        image_enc = self.ImageEncoder(item['image'])
        scores = platalea.loss.cosine_matrix(speech_enc, image_enc)
        loss =  platalea.loss.contrastive(scores, margin=self.config['margin_size'])
        return loss

    def embed_image(self, images):
        image = torch.utils.data.DataLoader(dataset=images, batch_size=32,
                                            shuffle=False,
                                            collate_fn=D.batch_image)
        image_e = []
        for i in image:
            image_e.append(self.ImageEncoder(i.to(_device)).detach().cpu().numpy())
        image_e = np.concatenate(image_e)
        return image_e

    def embed_audio(self, audios):
        audio = torch.utils.data.DataLoader(dataset=audios, batch_size=32,
                                            shuffle=False,
                                            collate_fn=D.batch_audio)
        audio_e = []
        for a, l in audio:
            audio_e.append(self.SpeechEncoder(a.to(_device), l.to(_device)).detach().cpu().numpy())
        audio_e = np.concatenate(audio_e)
        return audio_e


def cyclic_scheduler(optimizer, n_batches, max_lr, min_lr=1e-6):
    stepsize = n_batches * 4
    logging.info("Setting stepsize of {}".format(stepsize))
    lr_lambda = lambda iteration: (max_lr - min_lr)*(0.5 * (np.cos(np.pi * (1 + (3 - 1) / stepsize * iteration)) + 1))+min_lr
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    # lambda function which uses the cosine function to cycle the learning rate between the given min and max rates
    # the function operates between 1 and 3 (so the cos cycles from -1 to -1 ) normalise between 0 and 1 and then press between
    # min and max lr
    return scheduler


def dict_values_to_device(data, device):
    return {key: value.to(device) for key, value in data.items()}


def experiment(net, data, config):
    def val_loss():
        net.eval()  # switch to eval mode
        result = []
        for item in data['val']:
            item = dict_values_to_device(item, _device)
            result.append(net.cost(item).item())
        net.train()  # back to train mode
        return torch.tensor(result).mean()

    net.to(_device)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=1)
    scheduler = cyclic_scheduler(optimizer, len(data['train']), max_lr = config['max_lr'], min_lr = 1e-6)
    optimizer.zero_grad()

    with open("result.json", "w") as out:
        for epoch in range(1, config['epochs']+1):
            cost = Counter()
            for j, item in enumerate(data['train'], start=1): # check reshuffling
                item = dict_values_to_device(item, _device)
                loss = net.cost(item)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                cost += Counter({'cost': loss.item(), 'N':1})
                if j % 100 == 0:
                    logging.info("train {} {} {}".format(epoch, j, cost['cost']/cost['N']))
                else:
                    logging.debug("train {} {} {}".format(epoch, j, cost['cost']/cost['N']))
                if j % 400 == 0:
                    logging.info("valid {} {} {}".format(epoch, j, val_loss()))
                else:
                    logging.debug("valid {} {} {}".format(epoch, j, val_loss()))
            result = platalea.score.score(net, data['val'].dataset)
            result['epoch'] = epoch
            json.dump(result, out)
            print('', file=out, flush=True)
            logging.info("Saving model in net.{}.pt".format(epoch))
            torch.save(net, "net.{}.pt".format(epoch))


DEFAULT_CONFIG = dict(SpeechEncoder=dict(conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2, padding=0, bias=False),
                                         rnn= dict(input_size=64, hidden_size=1024, num_layers=4,
                                                   bidirectional=True, dropout=0),
                                         att= dict(in_size=2048, hidden_size=128)),
              ImageEncoder=dict(linear=dict(in_size=2048, out_size=2*1024), norm=True),
              margin_size=0.2)
