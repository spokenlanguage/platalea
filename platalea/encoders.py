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
    
class UnidiGRUStack(nn.Module):
    """Unidirectional GRU stack with separate GRU modules so that full intermediate states can be accessed."""
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0):
        super(UnidiGRUStack, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bottom = nn.GRU(input_size, hidden_size, num_layers=1,
                             bias=bias, batch_first=True, bidirectional=False,
                             dropout=dropout)
        self.layers = nn.ModuleList([nn.GRU(hidden_size, hidden_size, num_layers=1,
                                            bias=bias, batch_first=True,
                                            dropout=dropout, bidirectional=False)
                                     for i in range(num_layers-1) ])
    
    def forward(self, x):
        hidden = []
        last = []
        output, h_n = self.bottom(x)
        hidden.append(output)
        last.append(h_n)
        for rnn in self.layers:
            output, h_n = rnn(hidden[-1])
            hidden.append(output)
            last.append(h_n)
        return hidden[-1], torch.cat(last, dim=0)
    
class BidiGRUStack(nn.Module):
    """Bidirectional GRU stack with separate GRU modules so that full intermediate states can be accessed."""
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0):
        super(BidiGRUStack, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.l2r = UnidiGRUStack(input_size, hidden_size, num_layers=num_layers,
                             bias=bias, dropout=dropout)
        self.r2l = UnidiGRUStack(input_size, hidden_size, num_layers=num_layers,
                             bias=bias, dropout=dropout)

    def forward(self, x):
        # Assume we get a tensor
        try: 
            x_rev = x.flip(dims=[1])
            output_l2r, h_n_l2r = self.l2r(x)
            output_r2l, h_n_r2l = self.r2l(x_rev)
            output_r2l_rev = output_r2l.flip(dims=[1])
            output = torch.cat([output_l2r, output_r2l_rev], dim=2)
            h_n = torch.cat([h_n_l2r, h_n_r2l], dim=0)
        # We got a packed sequence
        except AttributeError: 
            x_rev = flip_packed(x)
            output_l2r, h_n_l2r = self.l2r(x)
            output_r2l, h_n_r2l = self.r2l(x_rev)
            output_r2l_rev = flip_packed(output_r2l)
            a, k = pad_packed_sequence(output_l2r, batch_first=True)
            b, l = pad_packed_sequence(output_r2l_rev, batch_first=True)
            output = pack_sequence(torch.cat([a, b], dim=2), enforce_sorted=False)
            h_n = torch.cat([h_n_l2r, h_n_r2l], dim=0)
        return output, h_n

class GRUStack(nn.Module):
    """GRU stack with separate GRU modules so that full intermediate states can be accessed."""
    def __init__(self, *args, bidirectional=False, batch_first=False, **kwargs):
        super(GRUStack, self).__init__()
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        if bidirectional:
            self.stack = BidiGRUStack(*args, **kwargs)
        else:
            self.stack = UnidiGRUStack(*args, **kwargs)
            
    def forward(self, x):
        if self.batch_first:
            return self.stack(x)
        else:
            output, h_n = self.stack(x.permute(1, 0, 2))
            return (output.permute(1, 0, 2), h_n)

def flip_packed(x):
    # Assumes batch first 
    z, l = pad_packed_sequence(x, batch_first=True, padding_value=0.0) 
    z_flip = [ z[i, :l[i]].flip(dims=[0]) for i in range(len(z)) ] 
    return pack_sequence(z_flip, enforce_sorted=False) 
