import torch
import torch.nn as nn
import numpy as np

class IntrospectGRU(nn.Module):
    """Wrapper around torch.nn.GRU which enables retrieving activations of intermediate layers."""
    def __init__(self, rnn):
        super(IntrospectGRU, self).__init__()
        self.rnn = rnn
        self.submodels = []
        for n in range(1, rnn.num_layers+1):
            model = nn.GRU(rnn.input_size, rnn.hidden_size,
                           num_layers=n, bias=rnn.bias,
                           batch_first=rnn.batch_first,
                           dropout=rnn.dropout, bidirectional=rnn.bidirectional)
            for key in model.__dict__['_parameters']:
                model.__dict__['_parameters'][key] = rnn.__dict__['_parameters'][key]
            self.submodels.append(model)

        
    def forward(self, x):
        return self.rnn(x)

    def introspect(self, x):
        out = []
        for model in self.submodels:
            hidden, _hn = model(x)
            out.append(hidden)
        return torch.stack(out, dim=0)

    
def inout(L, pad=0, ksize=6, stride=2):
    """Mapping from size of input to the size of the output of a 1D convolutional layer.
    https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
    """
    return np.floor( (L+2*pad-1*(ksize-1)-1)/stride + 1).astype(int)
