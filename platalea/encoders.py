import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
    
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
        stack = config['rnn'].pop('stack', False)
        if stack:
            self.RNN = GRUStack(batch_first=True, **rnn)
        else:
            self.RNN = nn.GRU(batch_first=True, **rnn)
        self.att = Attention(**att)
        
    def forward(self, input, l):
        x = self.Conv(input)
        # update the lengths to compensate for the convolution subsampling
        l = [int((y-(self.Conv.kernel_size[0]-self.Conv.stride[0]))/self.Conv.stride[0]) for y in l]
        # create a packed_sequence object. The padding will be excluded from the update step
        # thereby training on the original sequence length only
        x = nn.utils.rnn.pack_padded_sequence(x.transpose(2,1), l, batch_first=True, enforce_sorted=False)
        x, hx = self.RNN(x)
        # unpack again as at the moment only rnn layers except packed_sequence objects
        x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)
        x = nn.functional.normalize(self.att(x), p=2, dim=1)    
        return x
    
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
    
