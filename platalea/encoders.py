import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
import platalea.introspect
import logging

# Includes code adapted from  https://github.com/gchrupala/speech2image/blob/master/PyTorch/functions/encoders.py

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        linear = config['linear']
        self.norm = config['norm']
        self.linear_transform = nn.Linear(in_features = linear['in_size'], out_features = linear['out_size'])
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
        att  = config ['att'] 
        self.Conv = nn.Conv1d(**conv)
        self.RNN = nn.GRU(batch_first=True, **rnn)
        self.att = Attention(**att)
        
    def forward(self, input, l):
        x = self.Conv(input)
        # update the lengths to compensate for the convolution subsampling
        # l = [int((y-(self.Conv.kernel_size[0]-self.Conv.stride[0]))/self.Conv.stride[0]) for y in l]
        l = [ inout(L, pad=0, ksize=self.Conv.kernel_size[0], stride=self.Conv.stride[0]) for L in l ]
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = nn.utils.rnn.pack_padded_sequence(x.transpose(2,1), l, batch_first=True, enforce_sorted=False)
        x, hx = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
        return x

    def introspect(self, input, l):
        if not hasattr(self, 'IntrospectGRU'):
            logging.info("Creating IntrospectGRU wrapper")
            self.IntrospectGRU = platalea.introspect.IntrospectGRU(self.RNN)
        result = {}
        logging.info("Computing convolutional activations")
        result['conv'] = self.Conv(input).permute(0, 2, 1)
        l = [ inout(L, pad=0, ksize=self.Conv.kernel_size[0], stride=self.Conv.stride[0]) for L in l ]
        x = nn.utils.rnn.pack_padded_sequence(result['conv'], l, batch_first=True, enforce_sorted=False)

        logging.info("Computing full stack of RNN states")
        result['rnn'] = self.IntrospectGRU.introspect(x) 
        x, hx = self.RNN(x)
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)

        logging.info("Computing aggregated and normalized encoding")
        result['att'] = x
        return result
    
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Attention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.out = nn.Linear(hidden_size, in_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, input):
        # calculate the attention weights
        self.alpha = self.softmax(self.out(torch.tanh(self.hidden(input))))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(self.alpha * input, 1)
        # return the resulting embedding
        return x
    
def inout(L, pad=0, ksize=6, stride=2):
    """Mapping from size of input to the size of the output of a 1D convolutional layer.
    https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
    """
    return ( (L.float() + 2*pad - 1*(ksize-1) -1) / stride + 1).floor().long()
