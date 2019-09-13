import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from platalea.encoders import SpeechEncoder, ImageEncoder
import platalea.loss
from collections import Counter
import logging
from torch.optim import lr_scheduler

class SpeechImage(nn.Module):

    def __init__(self, config):
        super(SpeechImage, self).__init__()
        self.config = config
        self.SpeechEncoder = SpeechEncoder(config['SpeechEncoder'])
        self.ImageEncoder = ImageEncoder(config['ImageEncoder'])


    def cost(self, item):
        speech_enc = self.SpeechEncoder(item['audio'], item['audio_len'])
        image_enc = self.ImageEncoder(item['image'])
        scores = platalea.loss.cosine_matrix(speech_enc, image_enc) 
        loss =  platalea.loss.contrastive(scores, margin=self.config['margin_size'])
        return loss

def cyclic_scheduler(optimizer, n_batches, max_lr, min_lr=1e-6):
    stepsize = n_batches * 4
    logging.info("Setting stepsize of {}".format(stepsize))
    lr_lambda = lambda iteration: (max_lr - min_lr)*(0.5 * (np.cos(np.pi * (1 + (3 - 1) / stepsize * iteration)) + 1))+min_lr
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    # lambda function which uses the cosine function to cycle the learning rate between the given min and max rates
    # the function operates between 1 and 3 (so the cos cycles from -1 to -1 ) normalise between 0 and 1 and then press between
    # min and max lr   
    return scheduler

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
    optimizer = optim.Adam(net.parameters(), lr=1)
    scheduler = cyclic_scheduler(optimizer, len(data['train']), max_lr = config['max_lr'], min_lr = 1e-6)
    optimizer.zero_grad()
    
    with open("result.json", "w") as out:
        for epoch in range(1, config['epochs']+1):
            cost = Counter()
            for j, item in enumerate(data['train'], start=1): # check reshuffling
                item = {key: value.cuda() for key, value in item.items()}
                loss = net.cost(item)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                cost += Counter({'cost': loss.item(), 'N':1})
                if j % 100 == 0:
                    logging.info("train {} {} {}".format(epoch, j, cost['cost']/cost['N']))
                if j % 400 == 0:
                    logging.info("valid {} {} {}".format(epoch, j, val_loss()))
            logging.info("Saving model in net.{}.pt".format(epoch))
            torch.save(net, "net.{}.pt".format(epoch))
            
    
