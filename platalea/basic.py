from collections import Counter
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import wandb  # cloud logging

from platalea.encoders import SpeechEncoder, ImageEncoder
import platalea.loss
import platalea.dataset as D
import platalea.score
import platalea.hardware
import platalea.schedulers
from platalea.optimizers import create_optimizer
from platalea.schedulers import create_scheduler


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
        loss = platalea.loss.contrastive(scores, margin=self.config['margin_size'])
        return loss

    def embed_image(self, images):
        image = torch.utils.data.DataLoader(dataset=images, batch_size=32,
                                            shuffle=False,
                                            collate_fn=D.batch_image)
        image_e = []
        _device = platalea.hardware.device()
        for i in image:
            image_e.append(self.ImageEncoder(i.to(_device)).detach().cpu().numpy())
        image_e = np.concatenate(image_e)
        return image_e

    def embed_audio(self, audios):
        audio = torch.utils.data.DataLoader(dataset=audios, batch_size=32,
                                            shuffle=False,
                                            collate_fn=D.batch_audio)
        audio_e = []
        _device = platalea.hardware.device()
        for a, l in audio:
            audio_e.append(self.SpeechEncoder(a.to(_device), l.to(_device)).detach().cpu().numpy())
        audio_e = np.concatenate(audio_e)
        return audio_e


def dict_values_to_device(data, device):
    return {key: value.to(device) for key, value in data.items()}


def experiment(net, data, config,
               wandb_log=None,
               wandb_project="platalea",
               wandb_entity="spokenlanguage",
               wandb_mode=None):
    """

    :type wandb_log: (nested) dict with complete config to be logged by wandb
    """
    def val_loss(net):
        _device = platalea.hardware.device()
        net.eval()  # switch to eval mode
        result = []
        for item in data['val']:
            item = dict_values_to_device(item, _device)
            result.append(net.cost(item).item())
        net.train()  # back to train mode
        return torch.tensor(result).mean()

    if not wandb_log:
        wandb_log = config
    wandb.init(project=wandb_project, entity=wandb_entity, config=wandb_log,
               mode=wandb_mode)
    wandb.watch(net)

    _device = platalea.hardware.device()
    net.to(_device)
    net.train()
    net_parameters = net.parameters()
    optimizer = create_optimizer(config, net_parameters)
    scheduler = create_scheduler(config, optimizer, data)

    debug_logging_active = logging.getLogger().isEnabledFor(logging.DEBUG)

    loss_value = None
    results = []
    with open("result.json", "w") as out:
        for epoch in range(1, config['epochs']+1):
            cost = Counter()
            for j, item in enumerate(data['train'], start=1):  # check reshuffling
                wandb_step_output = {
                    "epoch": epoch,
                }

                item = dict_values_to_device(item, _device)
                loss = net.cost(item)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_value = loss.item()
                cost += Counter({'cost': loss_value, 'N': 1})
                average_loss = cost['cost'] / cost['N']

                # logging
                wandb_step_output["step loss"] = loss_value
                wandb_step_output["last_lr"] = scheduler.get_last_lr()[0]
                if j % config['loss_logging_interval'] == 0:
                    logging.info("train %d %d %f", epoch, j, average_loss)
                else:
                    if debug_logging_active:
                        logging.debug("train %d %d %f %f", epoch, j, average_loss, loss_value)
                if j % config['validation_interval'] == 0:
                    validation_loss = val_loss(net)
                    logging.info("valid %d %d %f", epoch, j, validation_loss)
                    wandb_step_output["validation loss"] = validation_loss
                else:
                    if debug_logging_active:
                        validation_loss = val_loss(net)
                        logging.debug("valid %d %d %f", epoch, j, validation_loss)
                        wandb_step_output["validation loss"] = validation_loss
                wandb.log(wandb_step_output)

            logging.info("Saving model in net.{}.pt".format(epoch))
            torch.save(net, "net.{}.pt".format(epoch))

            logging.info("Calculating and saving epoch score results")
            net.eval()
            result = platalea.score.score(net, data['val'].dataset)
            net.train()
            result['epoch'] = epoch
            result['average_loss'] = average_loss
            results.append(result)
            json.dump(result, out)
            print('', file=out, flush=True)
            wandb.log(result)

    return results


DEFAULT_CONFIG = dict(SpeechEncoder=dict(conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2, padding=0,
                                                   bias=False),
                                         rnn=dict(input_size=64, hidden_size=1024, num_layers=4,
                                                  bidirectional=True, dropout=0),
                                         att=dict(in_size=2048, hidden_size=128)),
                      ImageEncoder=dict(linear=dict(in_size=2048, out_size=2*1024), norm=True),
                      margin_size=0.2)
