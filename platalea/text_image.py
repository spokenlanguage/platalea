from collections import Counter
import json
import logging
import numpy as np
import torch
import torch.nn as nn

import platalea.schedulers
import platalea.dataset as D
from platalea.encoders import TextEncoder, ImageEncoder
import platalea.loss
import platalea.score
import platalea.hardware
from platalea.optimizers import create_optimizer
from platalea.schedulers import create_scheduler


class TextImage(nn.Module):
    def __init__(self, config):
        super(TextImage, self).__init__()
        self.config = config
        # Components can be pre-instantiated or configured through a dictionary
        if isinstance(config['TextEncoder'], nn.Module):
            self.TextEncoder = config['TextEncoder']
        else:
            self.TextEncoder = TextEncoder(config['TextEncoder'])
        if isinstance(config['ImageEncoder'], nn.Module):
            self.ImageEncoder = config['ImageEncoder']
        else:
            self.ImageEncoder = ImageEncoder(config['ImageEncoder'])

    def cost(self, item):
        text_enc = self.TextEncoder(item['text'], item['text_len'])
        image_enc = self.ImageEncoder(item['image'])
        scores = platalea.loss.cosine_matrix(text_enc, image_enc)
        loss = platalea.loss.contrastive(scores,
                                         margin=self.config['margin_size'])
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

    def embed_text(self, texts):
        texts = [D.caption2tensor(t) for t in texts]
        text = torch.utils.data.DataLoader(dataset=texts, batch_size=32,
                                           shuffle=False,
                                           collate_fn=D.batch_text)
        text_e = []
        _device = platalea.hardware.device()
        for t, l in text:
            text_e.append(self.TextEncoder(t.to(_device),
                                           l.to(_device)).detach().cpu().numpy())
        text_e = np.concatenate(text_e)
        return text_e


def experiment(net, data, config):
    _device = platalea.hardware.device()

    def val_loss():
        net.eval()
        result = []
        for item in data['val']:
            item = {key: value.to(_device) for key, value in item.items()}
            result.append(net.cost(item).item())
        net.train()
        return torch.tensor(result).mean()

    net.to(_device)
    net.train()
    net_parameters = net.parameters()
    optimizer = create_optimizer(config, net_parameters)
    scheduler = create_scheduler(config, optimizer, data)

    results = []
    with open("result.json", "w") as out:
        for epoch in range(1, config['epochs']+1):
            cost = Counter()
            average_loss = None
            for j, item in enumerate(data['train'], start=1):
                item = {key: value.to(_device) for key, value in item.items()}
                loss = net.cost(item)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                cost += Counter({'cost': loss.item(), 'N': 1})
                average_loss = cost['cost'] / cost['N']
                if j % config['loss_logging_interval'] == 0:
                    logging.info("train {} {} {}".format(
                        epoch, j, average_loss))
                if j % config['validation_interval'] == 0:
                    logging.info("valid {} {} {}".format(epoch, j, val_loss()))
            result = platalea.score.score_text_image(net, data['val'].dataset)
            result['average_loss'] = average_loss
            result['epoch'] = epoch
            results.append(result)
            json.dump(result, out)
            print('', file=out, flush=True)
            logging.info("Saving model in net.{}.pt".format(epoch))
            torch.save(net, "net.{}.pt".format(epoch))
    return results


def get_default_config(hidden_size_factor=1024):
    return dict(
        TextEncoder=dict(
            emb=dict(num_embeddings=len(D.tokenizer.classes_),
                     embedding_dim=128),
            rnn=dict(input_size=128, hidden_size=hidden_size_factor, num_layers=2,
                     bidirectional=True, dropout=0),
            att=dict(in_size=hidden_size_factor * 2, hidden_size=128)),
        ImageEncoder=dict(
            linear=dict(in_size=2048, out_size=hidden_size_factor * 2),
            norm=True),
        margin_size=0.2)
