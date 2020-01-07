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


class SpeechEncoder(nn.Module):
    def __init__(self, config):
        super(SpeechEncoder, self).__init__()
        conv = config['conv']
        rnn  = config['rnn']
        att  = config.get('att', None)
        self.Conv = nn.Conv1d(**conv)
        rnn_layer_type = config.get('rnn_layer_type', nn.GRU)
        self.RNN = rnn_layer_type(batch_first=True, **rnn)
        if att is not None:
            self.att = Attention(**att)
        else:
            self.att = None

    def forward(self, input, l):
        x = self.Conv(input)
        # update the lengths to compensate for the convolution subsampling
        # l = [int((y-(self.Conv.kernel_size[0]-self.Conv.stride[0]))/self.Conv.stride[0]) for y in l]
        l = inout(self.Conv, l)
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = nn.utils.rnn.pack_padded_sequence(x.transpose(2, 1), l, batch_first=True, enforce_sorted=False)
        x, _ = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

    def introspect(self, input, l):
        if not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN)
        result = {}

        # Computing convolutional activations
        conv = self.Conv(input).permute(0, 2, 1)
        l = inout(self.Conv, l)
        result['conv'] = [conv[i, :l[i], :] for i in range(len(conv))]

        # Computing full stack of RNN states
        conv_padded = nn.utils.rnn.pack_padded_sequence(
            conv, l, batch_first=True, enforce_sorted=False)
        rnn = self.IntrospectRNN.introspect(conv_padded)
        for layer in range(self.RNN.num_layers):
            result['rnn{}'.format(layer)] = [rnn[i, layer, :l[i], :] for i in range(len(rnn))]

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

    def forward(self, input, l):
        x = input
        for conv in self.Conv:
            x = conv(x)
            # update the lengths to compensate for the convolution subsampling
            l = inout(conv, l)
        # create a packed_sequence object. The padding will be excluded from
        # the update step thereby training on the original sequence length only
        x = nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1), l, batch_first=True, enforce_sorted=False)
        x, _ = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence
        # objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

    def introspect(self, input, l):
        if not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN)
        result = {}

        # Computing convolutional activations
        x = input
        for i, conv in enumerate(self.Conv):
            x = conv(x)
            # update the lengths to compensate for the convolution subsampling
            l = inout(conv, l)
            x_perm = x.permute(0, 2, 1)
            result['conv{}'.format(i)] = [x_perm[i, :l[i], :] for i in range(len(x_perm))]

        # Computing full stack of RNN states
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1), l, batch_first=True, enforce_sorted=False)
        rnn = self.IntrospectRNN.introspect(x_packed)
        for layer in range(self.RNN.num_layers):
            result['rnn{}'.format(layer)] = [rnn[i, layer, :l[i], :] for i in range(len(rnn))]

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

    def forward(self, input, l):
        x = F.relu(self.Conv0(input.unsqueeze(1)))
        l = inout(self.Conv0, l)
        x = F.relu(self.Conv1(x))
        l = inout(self.Conv1, l)
        x = self.MaxPool(x)
        l = inout(self.MaxPool, l)
        x = F.relu(self.Conv2(x))
        l = inout(self.Conv2, l)
        x = F.relu(self.Conv3(x))
        l = inout(self.Conv3, l)
        x = self.MaxPool(x)
        l = inout(self.MaxPool, l)
        # create a packed_sequence object. The padding will be excluded from
        # the update step thereby training on the original sequence length only
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1), l, batch_first=True, enforce_sorted=False)
        x, _ = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence
        # objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

    def introspect(self, input, l):
        if not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN)
        result = {}

        # Computing convolutional activations
        x = F.relu(self.Conv0(input.unsqueeze(1)))
        l = inout(self.Conv0, l)
        result['conv0'] = [x[i, :l[i], :] for i in range(len(x))]
        x = F.relu(self.Conv1(x))
        l = inout(self.Conv1, l)
        x = self.MaxPool(x)
        l = inout(self.MaxPool, l)
        result['conv1'] = [x[i, :l[i], :] for i in range(len(x))]
        x = F.relu(self.Conv2(x))
        l = inout(self.Conv2, l)
        result['conv2'] = [x[i, :l[i], :] for i in range(len(x))]
        x = F.relu(self.Conv3(x))
        l = inout(self.Conv3, l)
        x = self.Maxpool(x)
        l = inout(self.MaxPool, l)
        result['conv3'] = [x[i, :l[i], :] for i in range(len(x))]

        # Computing full stack of RNN states
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1), l, batch_first=True, enforce_sorted=False)
        rnn = self.IntrospectRNN.introspect(x_packed)
        for layer in range(self.RNN.num_layers):
            result['rnn{}'.format(layer)] = [rnn[i, layer, :l[i], :] for i in range(len(rnn))]

        # Computing aggregated and normalized encoding
        x, _ = self.RNN(x_packed)
        x, _lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
            result['att'] = list(x)
        return result


def inout(layer, L):
    """Mapping from size of input to the size of the output of a 1D
    convolutional layer.
    https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
    """
    if type(layer) == nn.Conv1d:
        fn = lambda x: x[0]
    elif type(layer) == nn.Conv2d:
        fn = lambda x: x[1]
    elif type(layer) == nn.MaxPool1d or type(layer) == nn.MaxPool2d:
        fn = lambda x: x
    else:
        raise NotImplementedError
    pad = fn(layer.padding)
    ksize = fn(layer.kernel_size)
    stride = fn(layer.stride)
    dilation = fn(layer.dilation)
    L = ((L.float() + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1)
    return L.floor().long()
