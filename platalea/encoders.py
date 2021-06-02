from collections import OrderedDict
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from platalea.attention import Attention
import platalea.hardware
import platalea.introspect
from platalea.vq import VQEmbeddingEMA

# Includes code adapted from
# https://github.com/gchrupala/speech2image/blob/master/PyTorch/functions/encoders.py


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
        x = nn.utils.rnn.pack_padded_sequence(x, length.cpu(), batch_first=True,
                                              enforce_sorted=False)
        x, _ = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence
        # objects
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.att is not None:
            x = nn.functional.normalize(self.att(x), p=2, dim=1)
        return x

    def introspect(self, text, length):
        if not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN)
        result = {}

        # Computing convolutional activations
        embed = self.Embed(text)
        result['embed'] = [embed[i, :length[i], :] for i in range(len(embed))]

        # Computing full stack of RNN states
        embed_padded = nn.utils.rnn.pack_padded_sequence(
            embed, length.cpu(), batch_first=True, enforce_sorted=False)
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
            x.transpose(2, 1), length.cpu(), batch_first=True, enforce_sorted=False)
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
            conv, length.cpu(), batch_first=True, enforce_sorted=False)
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


def generate_padding_mask(batch_size, lengths, max_len=None):
    # when a value is True, the corresponding value on the attention layer will be ignored
    # (https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward)
    if max_len is None:
        max_len = max(lengths)
    mask = torch.zeros((batch_size, max_len), dtype=bool)
    for ix, l in enumerate(lengths):
        mask[ix, l:] = True
    return mask


class SpeechEncoderTransformer(nn.Module):
    def __init__(self, config):
        super(SpeechEncoderTransformer, self).__init__()

        conv = config['conv']
        self.Conv = nn.Conv1d(**conv)

        trafo = config['trafo']
        num_layers = trafo.pop('num_encoder_layers')

        def default_transformer_layer(**config):
            return nn.TransformerEncoder(nn.TransformerEncoderLayer(**config), num_layers)
        trafo_layer_type = config.get('trafo_layer_type', default_transformer_layer)
        self.Transformer = trafo_layer_type(**trafo)

        upsample = config['upsample']
        if trafo['d_model'] == conv['out_channels']:
            self.scale_conv_to_trafo = nn.Identity()
        else:
            self.scale_conv_to_trafo = nn.Linear(in_features=conv['out_channels'],
                                                 out_features=trafo['d_model'], **upsample)

        att = config.get('att', None)
        if att is not None:
            self.att = Attention(**att)
        else:
            self.att = None

    def forward(self, src, lengths):
        x = self.Conv(src)

        # update the lengths to compensate for the convolution subsampling
        lengths = inout(self.Conv, lengths)

        # # source sequence dimension must be first (but is last in input),
        # # batch dimension in the middle (was first in input), feature dimension last
        # x = x.permute(2, 0, 1)
        x = x.permute(0, 2, 1)

        x = self.scale_conv_to_trafo(x)

        x = x.permute(1, 0, 2)

        mask = generate_padding_mask(x.size()[1], lengths).to(platalea.hardware.device())
        x = torch.utils.checkpoint.checkpoint(lambda a, b: self.Transformer(a, src_key_padding_mask=b),
                                              x, mask)

        x = x.transpose(1, 0)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)

        return x

    # EGP TODO: not sure how to handle introspect, discuss; what should it do and how does that translate to the trafo?
    def introspect(self, input, lengths):
        raise NotImplementedError


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
            x.transpose(2, 1), length.cpu(), batch_first=True, enforce_sorted=False)
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
            x.transpose(2, 1), length.cpu(), batch_first=True, enforce_sorted=False)
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
            x.transpose(2, 1), length.cpu(), batch_first=True, enforce_sorted=False)
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
            x.transpose(2, 1), length.cpu(), batch_first=True, enforce_sorted=False)
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
        # FIXME. Avoid negative lengths which `inout` returns in some cases.
        length = length.clamp(min=1)
        # Create a packed_sequence object. The padding will be excluded from
        # the update step thereby training on the original sequence length
        # only.  Expecting a SpeechEncoderTop to unpack the sequence
        x = nn.utils.rnn.pack_padded_sequence(
            x.transpose(2, 1), length.cpu(), batch_first=True, enforce_sorted=False)
        if self.RNN is not None:
            x, _ = self.RNN(x)
        return x

    def introspect(self, input, length):
        if self.RNN is not None and not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN).to(next(self.RNN.parameters()))
        result = {}

        # Computing convolutional activations
        conv = self.Conv(input).permute(0, 2, 1)
        length = inout(self.Conv, length)
        result['conv'] = [conv[i, :length[i], :] for i in range(len(conv))]
        # Computing full stack of RNN states
        x = nn.utils.rnn.pack_padded_sequence(
            conv, length.cpu(), batch_first=True, enforce_sorted=False)
        if self.RNN is not None:
            rnn = self.IntrospectRNN.introspect(x)
            for l in range(self.RNN.num_layers):
                name = 'rnn_bottom{}'.format(l)
                result[name] = [rnn[i, l, :length[i], :] for i in range(len(rnn))]
            x, _ = self.RNN(x)

        return result, length


class SpeechEncoderMiddle(nn.Module):
    def __init__(self, config):
        super(SpeechEncoderMiddle, self).__init__()
        # Potential RNN layer(s)
        rnn = config.get('rnn', None)
        if rnn is None:
            self.RNN = None
        else:
            rnn_layer_type = config.get('rnn_layer_type', nn.GRU)
            self.RNN = rnn_layer_type(batch_first=True, **rnn)

    def forward(self, x):
        # Expecting packed sequence
        if self.RNN is not None:
            x, _ = self.RNN(x)
        return x

    def introspect(self, input, length):
        if self.RNN is not None and not hasattr(self, 'IntrospectRNN'):
            logging.info("Creating IntrospectRNN wrapper")
            self.IntrospectRNN = platalea.introspect.IntrospectRNN(self.RNN).to(next(self.RNN.parameters()))
        result = {}
        # Computing full stack of RNN states
        x = nn.utils.rnn.pack_padded_sequence(
            input, length.cpu(), batch_first=True, enforce_sorted=False)
        if self.RNN is not None:
            rnn = self.IntrospectRNN.introspect(x)
            for l in range(self.RNN.num_layers):
                name = 'rnn_bottom{}'.format(l)
                result[name] = [rnn[i, l, :length[i], :] for i in range(len(rnn))]
            x, _ = self.RNN(x)

        return result, length


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


class SpeechEncoderVQ(nn.Module):
    def __init__(self, config):
        super(SpeechEncoderVQ, self).__init__()
        self.Bottom = SpeechEncoderBottom(config['SpeechEncoderBottom'])
        self.Codebook = VQEmbeddingEMA(config['VQEmbedding']['num_codebook_embeddings'],
                                       config['VQEmbedding']['embedding_dim'], jitter=config['VQEmbedding']['jitter'])
        self.Top = SpeechEncoderTop(config['SpeechEncoderTop'])

    def forward(self, input, length):
        return self.Top(self.Codebook(self.Bottom(input, length))['quantized'])

    def introspect(self, input, length):

        x = self.Bottom(input, length)
        # get intro and updated length
        bottom, length = self.Bottom.introspect(input, length)
        # FIXME probably not needed to run both forward and introspect?
        x = self.Codebook(x)
        codebook = [xi[:l, :] for xi, l in zip(x['one_hot'], length)]
        top = self.Top.introspect(x['quantized'], length)
        result = {**bottom, **dict(codebook=codebook), **top}
        return result


class SpeechEncoderVQ2(nn.Module):
    def __init__(self, config):
        super(SpeechEncoderVQ2, self).__init__()
        self.Bottom = SpeechEncoderBottom(config['SpeechEncoderBottom'])
        self.Codebook1 = VQEmbeddingEMA(config['VQEmbedding1']['num_codebook_embeddings'],
                                        config['VQEmbedding1']['embedding_dim'], jitter=config['VQEmbedding1']['jitter'])
        self.Middle = SpeechEncoderMiddle(config['SpeechEncoderMiddle'])
        self.Codebook2 = VQEmbeddingEMA(config['VQEmbedding2']['num_codebook_embeddings'],
                                        config['VQEmbedding2']['embedding_dim'], jitter=config['VQEmbedding2']['jitter'])
        self.Top = SpeechEncoderTop(config['SpeechEncoderTop'])

    def forward(self, input, length):
        return self.Top(self.Codebook2(self.Middle(self.Codebook1(self.Bottom(input, length))['quantized']))['quantized'])

    def introspect(self, input, length):

        x = self.Bottom(input, length)
        # get intro and updated length

        bottom, length = self.Bottom.introspect(input, length)

        x = self.Codebook1(x)
        codebook1 = [xi[:l, :] for xi, l in zip(x['one_hot'], length)]

        middle, length = self.Middle.introspect(x['quantized'], length)

        x = self.Codebook2(x)
        codebook2 = [xi[:l, :] for xi, l in zip(x['one_hot'], length)]

        top = self.Top.introspect(x['quantized'], length)

        result = {**bottom, **dict(codebook1=codebook1), **middle, **dict(codebook2=codebook2), **top}
        return result


def inout(layer, input_length):
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
    input_length = ((input_length.float() + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1)
    if maxpool and layer.ceil_mode:
        input_length = input_length.ceil()
    else:
        input_length = input_length.floor()
    output_length = input_length.long().clamp(min=0)
    return output_length
