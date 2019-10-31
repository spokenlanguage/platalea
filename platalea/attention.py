import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    def __init__(self, in_size):
        super(LinearAttention, self).__init__()
        self.out = nn.Linear(in_size, 1)
        nn.init.orthogonal_(self.out.weight.data)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        # calculate the scalar attention weights
        self.alpha = self.softmax(self.out(input))
        # apply the scalar weights to the input and sum over all timesteps
        x = (self.alpha.expand_as(input) * input).sum(dim=1)
        # return the resulting embedding
        return x


class ScalarAttention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(ScalarAttention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.out = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        # calculate the scalar attention weights
        self.alpha = self.softmax(self.out(torch.tanh(self.hidden(input))))
        # apply the scalar weights to the input and sum over all timesteps
        x = (self.alpha.expand_as(input) * input).sum(dim=1)
        # return the resulting embedding
        return x


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Attention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.out = nn.Linear(hidden_size, in_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        # calculate the attention weights
        self.alpha = self.softmax(self.out(torch.tanh(self.hidden(input))))
        # apply the weights to the input and sum over all timesteps
        x = torch.sum(self.alpha * input, 1)
        # return the resulting embedding
        return x


class BahdanauAttention(nn.Module):
    '''
    Attention mechanism following Bahdanau et al. (2015)
    [https://arxiv.org/abs/1409.0473]
    '''
    def __init__(self, in_size_enc, in_size_state, hidden_size):
        super(BahdanauAttention, self).__init__()

        self.U_a = nn.Linear(in_size_enc, hidden_size, bias=False)
        self.W_a = nn.Linear(in_size_state, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # Calculate energies for each encoder output
        hidden = hidden.permute(1, 0, 2)  # B x S x N
        attn_energies = self.W_a(hidden) + self.U_a(encoder_outputs)
        attn_energies = torch.tanh(attn_energies)
        attn_energies = self.v_a(attn_energies)

        # Normalize energies to weights in range 0 to 1,
        return F.softmax(attn_energies, dim=1)
