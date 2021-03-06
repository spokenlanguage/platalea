import torch
import torch.nn as nn


class IntrospectRNN(nn.Module):
    """Wrapper around torch.nn.GRU/LSTM which enables retrieving activations
    of intermediate layers."""
    def __init__(self, rnn):
        super(IntrospectRNN, self).__init__()
        self.rnn = rnn
        self.submodels = []
        for n in range(1, rnn.num_layers+1):
            model = type(self.rnn)(rnn.input_size, rnn.hidden_size,
                                   num_layers=n, bias=rnn.bias,
                                   batch_first=rnn.batch_first,
                                   dropout=rnn.dropout,
                                   bidirectional=rnn.bidirectional)
            for key in model.__dict__['_parameters']:
                model.__dict__['_parameters'][key] = rnn.__dict__['_parameters'][key]
            self.submodels.append(model.cuda())

    def forward(self, x):
        return self.rnn(x)

    def introspect(self, x):
        out = []
        for model in self.submodels:
            model.flatten_parameters()
            hidden, _ = model(x)
            out.append(hidden)
        try:
            result = torch.stack(out, dim=0)
        except TypeError:  # We got a PackedSequence
            out = [nn.utils.rnn.pad_packed_sequence(out_i, batch_first=True)[0] for out_i in out]
            result = torch.stack(out, dim=0)
        return result.permute(1, 0, 2, 3)
