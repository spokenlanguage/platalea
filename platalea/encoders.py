from collections import OrderedDict
import logging
import torch.nn as nn
import torch.nn.functional as F

import platalea.introspect
from platalea.attention import Attention

# Includes code adapted from  https://github.com/gchrupala/speech2image/blob/master/PyTorch/functions/encoders.py


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        linear = config['linear']
        self.norm = config['norm']
        self.linear_transform = nn.Linear(in_features=linear['in_size'],
                                          out_features=linear['out_size'])
        nn.init.xavier_uniform_(self.linear_transform.weight.data)

    def forward(self, input):
        x = self.linear_transform(input)
        if self.norm:
            return nn.functional.normalize(x, p=2, dim=1)
        else:
            return x


class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        emb = config['emb']
        rnn = config['rnn']
        att = config.get('att', None)
        self.Embed = nn.Embedding(**emb)
        rnn_layer_type = config.get('rnn_layer_type', nn.GRU)
        self.RNN = rnn_layer_type(batch_first=True, **rnn)
        if att is not None:
            self.att = Attention(**att)
        else:
            self.att = None

    def forward(self, text, length):
        x = self.Embed(text)
        # create a packed_sequence object. The padding will be excluded from
        # the update step thereby training on the original sequence length only
        x = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True,
                                              enforce_sorted=False)
        x, _ = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence
        # objects
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

    def introspect(self, text, length):  # TODO: PLEASE REVIEW input -> text RENAME!
        if not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN)
        result = {}

        # Computing convolutional activations
        embed = self.Embed(text)
        result['embed'] = [embed[i, :length[i], :] for i in range(len(embed))]

        # Computing full stack of RNN states
        embed_padded = nn.utils.rnn.pack_padded_sequence(
            embed, length, batch_first=True, enforce_sorted=False)
        rnn = self.IntrospectRNN.introspect(embed_padded)
        for l in range(self.RNN.num_layers):
            name = 'rnn{}'.format(l)
            result[name] = [rnn[i, l, :length[i], :] for i in range(len(rnn))]

        # Computing aggregated and normalized encoding
        x, _ = self.RNN(embed_padded)
        x, _lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
            result['att'] = list(x)
        return result


class SpeechEncoder(nn.Module):
    def __init__(self, config):
        super(SpeechEncoder, self).__init__()
        conv = config['conv']
        rnn = config['rnn']
        att = config.get('att', None)
        self.Conv = nn.Conv1d(**conv)
        rnn_layer_type = config.get('rnn_layer_type', nn.GRU)
        self.RNN = rnn_layer_type(batch_first=True, **rnn)
        if att is not None:
            self.att = Attention(**att)
        else:
            self.att = None

    def forward(self, input, length):
        x = self.Conv(input)
        # update the lengths to compensate for the convolution subsampling
        length = inout(self.Conv, length)
        # create a packed_sequence object. The padding will be excluded from
        # the update step thereby training on the original sequence length only
        x = nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1), length, batch_first=True, enforce_sorted=False)
        x, _ = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence
        # objects
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

    def introspect(self, input, length):
        if not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN)
        result = {}

        # Computing convolutional activations
        conv = self.Conv(input).permute(0, 2, 1)
        length = inout(self.Conv, length)
        result['conv'] = [conv[i, :length[i], :] for i in range(len(conv))]

        # Computing full stack of RNN states
        conv_padded = nn.utils.rnn.pack_padded_sequence(
            conv, length, batch_first=True, enforce_sorted=False)
        rnn = self.IntrospectRNN.introspect(conv_padded)
        for l in range(self.RNN.num_layers):
            name = 'rnn{}'.format(l)
            result[name] = [rnn[i, l, :length[i], :] for i in range(len(rnn))]

        # Computing aggregated and normalized encoding
        x, _ = self.RNN(conv_padded)
        x, _lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
            result['att'] = list(x)
        return result


class SpeechEncoderMultiConv(nn.Module):
    def __init__(self, config):
        super(SpeechEncoderMultiConv, self).__init__()
        conv = config['conv']
        rnn = config['rnn']
        att = config.get('att', None)
        layers = OrderedDict()
        for i, c in enumerate(conv):
            layers['conv{}'.format(i)] = nn.Conv1d(**c)
        self.Conv = nn.Sequential(layers)
        rnn_layer_type = config.get('rnn_layer_type', nn.GRU)
        self.RNN = rnn_layer_type(batch_first=True, **rnn)
        if att is not None:
            self.att = Attention(**att)
        else:
            self.att = None

    def forward(self, input, length):
        x = input
        for conv in self.Conv:
            x = conv(x)
            # update the lengths to compensate for the convolution subsampling
            length = inout(conv, length)
        # create a packed_sequence object. The padding will be excluded from
        # the update step thereby training on the original sequence length only
        x = nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1), length, batch_first=True, enforce_sorted=False)
        x, _ = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence
        # objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

    def introspect(self, input, length):
        if not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN)
        result = {}

        # Computing convolutional activations
        x = input
        for i, conv in enumerate(self.Conv):
            x = conv(x)
            # update the lengths to compensate for the convolution subsampling
            length = inout(conv, length)
            xp = x.permute(0, 2, 1)
            name = 'conv{}'.format(i)
            result[name] = [xp[i, :length[i], :] for i in range(len(xp))]

        # Computing full stack of RNN states
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1), length, batch_first=True, enforce_sorted=False)
        rnn = self.IntrospectRNN.introspect(x_packed)
        for l in range(self.RNN.num_layers):
            name = 'rnn{}'.format(l)
            result[name] = [rnn[i, l, :length[i], :] for i in range(len(rnn))]

        # Computing aggregated and normalized encoding
        x, _ = self.RNN(x_packed)
        x, _lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
            result['att'] = list(x)
        return result


class SpeechEncoderVGG(nn.Module):
    def __init__(self, config):
        super(SpeechEncoderVGG, self).__init__()
        rnn = config['rnn']
        att = config.get('att', None)
        self.Conv0 = nn.Conv2d(1, 64, 3, 1, 1)
        self.Conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.Conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.Conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.MaxPool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        rnn_layer_type = config.get('rnn_layer_type', nn.GRU)
        self.RNN = rnn_layer_type(batch_first=True, **rnn)
        if att is not None:
            self.att = Attention(**att)
        else:
            self.att = None

    def forward(self, input, length):
        x = F.relu(self.Conv0(input.unsqueeze(1)))
        length = inout(self.Conv0, length)
        x = F.relu(self.Conv1(x))
        length = inout(self.Conv1, length)
        x = self.MaxPool(x)
        length = inout(self.MaxPool, length)
        x = F.relu(self.Conv2(x))
        length = inout(self.Conv2, length)
        x = F.relu(self.Conv3(x))
        length = inout(self.Conv3, length)
        x = self.MaxPool(x)
        length = inout(self.MaxPool, length)
        # create a packed_sequence object. The padding will be excluded from
        # the update step thereby training on the original sequence length only
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1), length, batch_first=True, enforce_sorted=False)
        x, _ = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence
        # objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

    def introspect(self, input, length):
        if not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN)
        result = {}

        # Computing convolutional activations
        x = F.relu(self.Conv0(input.unsqueeze(1)))
        length = inout(self.Conv0, length)
        result['conv0'] = [x[i, :length[i], :] for i in range(len(x))]
        x = F.relu(self.Conv1(x))
        length = inout(self.Conv1, length)
        x = self.MaxPool(x)
        length = inout(self.MaxPool, length)
        result['conv1'] = [x[i, :length[i], :] for i in range(len(x))]
        x = F.relu(self.Conv2(x))
        length = inout(self.Conv2, length)
        result['conv2'] = [x[i, :length[i], :] for i in range(len(x))]
        x = F.relu(self.Conv3(x))
        length = inout(self.Conv3, length)
        x = self.Maxpool(x)
        length = inout(self.MaxPool, length)
        result['conv3'] = [x[i, :length[i], :] for i in range(len(x))]

        # Computing full stack of RNN states
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1), length, batch_first=True, enforce_sorted=False)
        rnn = self.IntrospectRNN.introspect(x_packed)
        for l in range(self.RNN.num_layers):
            name = 'rnn{}'.format(l)
            result[name] = [rnn[i, l, :length[i], :] for i in range(len(rnn))]

        # Computing aggregated and normalized encoding
        x, _ = self.RNN(x_packed)
        x, _lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
            result['att'] = list(x)
        return result


class SpeechEncoderBottom(nn.Module):
    def __init__(self, config):
        super(SpeechEncoderBottom, self).__init__()
        # Convolutional layer
        conv = config['conv']
        self.Conv = nn.Conv1d(**conv)
        # Potential RNN layer(s)
        rnn = config.get('rnn', None)
        if rnn is None:
            self.RNN = None
        else:
            rnn_layer_type = config.get('rnn_layer_type', nn.GRU)
            self.RNN = rnn_layer_type(batch_first=True, **rnn)

    def forward(self, input, length):
        x = self.Conv(input)
        # Update the lengths to compensate for the convolution subsampling
        length = inout(self.Conv, length)
        # Create a packed_sequence object. The padding will be excluded from
        # the update step thereby training on the original sequence length
        # only.  Expecting a SpeechEncoderTop to unpack the sequence
        x = nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1), length, batch_first=True, enforce_sorted=False)
        if self.RNN is not None:
            x, _ = self.RNN(x)
        return x

    def introspect(self, input, length):
        if self.RNN is not None and not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN)
        result = {}

        # Computing convolutional activations
        conv = self.Conv(input).permute(0, 2, 1)
        length = inout(self.Conv, length)
        result['conv'] = [conv[i, :length[i], :] for i in range(len(conv))]

        # Computing full stack of RNN states
        x = nn.utils.rnn.pack_padded_sequence(
            conv, length, batch_first=True, enforce_sorted=False)
        if self.RNN is not None:
            rnn = self.IntrospectRNN.introspect(x)
            for l in range(self.RNN.num_layers):
                name = 'rnn_bottom{}'.format(l)
                result[name] = [rnn[i, l, :length[i], :] for i in range(len(rnn))]
            x, _ = self.RNN(x)

        return x, result


class SpeechEncoderTop(nn.Module):
    def __init__(self, config):
        super(SpeechEncoderTop, self).__init__()
        rnn = config.get('rnn', None)
        if rnn is None:
            self.RNN = None
        else:
            rnn_layer_type = config.get('rnn_layer_type', nn.GRU)
            self.RNN = rnn_layer_type(batch_first=True, **rnn)
        att = config.get('att', None)
        if att is not None:
            self.att = Attention(**att)
        else:
            self.att = None

    def forward(self, x):
        # Expecting packed sequence
        if self.RNN is not None:
            x, _ = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence
        # objects
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

    def introspect(self, x, length):
        if self.RNN is not None and not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN)
        result = {}

        # Computing full stack of RNN states
        if self.RNN is not None:
            rnn = self.IntrospectRNN.introspect(x)
            for l in range(self.RNN.num_layers):
                name = 'rnn_top{}'.format(l)
                result[name] = [rnn[i, l, :length[i], :] for i in range(len(rnn))]
            x, _ = self.RNN(x)

        # Computing aggregated and normalized encoding
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
            result['att'] = list(x)
        return result


class SpeechEncoderSplit(nn.Module):
    def __init__(self, config):
        super(SpeechEncoderSplit, self).__init__()
        # Components can be pre-instantiated or configured through a dictionary
        if isinstance(config['SpeechEncoderBottom'], nn.Module):
            self.Bottom = config['SpeechEncoderBottom']
        else:
            self.Bottom = SpeechEncoderBottom(config['SpeechEncoderBottom'])
        if isinstance(config['SpeechEncoderTop'], nn.Module):
            self.Top = config['SpeechEncoderTop']
        else:
            self.Top = SpeechEncoderTop(config['SpeechEncoderTop'])

    def forward(self, input, length):
        return self.Top(self.Bottom(input, length))

    def introspect(self, input, length):
        x, result = self.Bottom(input)
        result.update(self.Top(x))
        return result


def inout(layer, L):
    """Mapping from size of input to the size of the output of a 1D
    convolutional layer.
    https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
    """
    maxpool = False
    if type(layer) == nn.Conv1d:
        def fn(x):
            return x[0]
    elif type(layer) == nn.Conv2d:
        def fn(x):
            return x[1]
    elif type(layer) == nn.MaxPool1d or type(layer) == nn.MaxPool2d:
        def fn(x):
            return x
        maxpool = True
    else:
        raise NotImplementedError
    pad = fn(layer.padding)
    ksize = fn(layer.kernel_size)
    stride = fn(layer.stride)
    dilation = fn(layer.dilation)
    L = ((L.float() + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1)
    if maxpool and layer.ceil_mode:
        L = L.ceil()
    else:
        L = L.floor()
    return L.long()
