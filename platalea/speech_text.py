import numpy as np
import torch
import torch.nn as nn

import platalea.dataset as D
from platalea.encoders import TextEncoder, SpeechEncoder
import platalea.loss
import platalea.score
import platalea.hardware


class SpeechText(nn.Module):
    def __init__(self, config):
        super(SpeechText, self).__init__()
        self.config = config
        # Components can be pre-instantiated or configured through a dictionary
        if isinstance(config['SpeechEncoder'], nn.Module):
            self.SpeechEncoder = config['SpeechEncoder']
        else:
            self.SpeechEncoder = SpeechEncoder(config['SpeechEncoder'])
        if isinstance(config['TextEncoder'], nn.Module):
            self.TextEncoder = config['TextEncoder']
        else:
            self.TextEncoder = TextEncoder(config['TextEncoder'])

    def cost(self, item):
        speech_enc = self.SpeechEncoder(item['audio'], item['audio_len'])
        text_enc = self.TextEncoder(item['text'], item['text_len'])
        scores = platalea.loss.cosine_matrix(speech_enc, text_enc)
        loss = platalea.loss.contrastive(scores,
                                         margin=self.config['margin_size'])
        return loss

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

    def embed_audio(self, audios):
        audio = torch.utils.data.DataLoader(dataset=audios, batch_size=32,
                                            shuffle=False,
                                            collate_fn=D.batch_audio)
        audio_e = []
        _device = platalea.hardware.device()
        for a, l in audio:
            audio_e.append(self.SpeechEncoder(a.to(_device),
                                              l.to(_device)).detach().cpu().numpy())
        audio_e = np.concatenate(audio_e)
        return audio_e
