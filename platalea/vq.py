# Code adapted from: https://github.com/bshall/ZeroSpeech/blob/abe02e66bfd1ebd08d88cac0fe7c14b672a7e711/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Jitter(nn.Module):
    def __init__(self, p):
        super().__init__()
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer("prob", prob)

    def forward(self, x):
        if not self.training:
            return x
        else:
            batch_size, sample_size, channels = x.size()

            dist = Categorical(self.prob)
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)

            x = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        return x


class VQEmbeddingEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5, jitter=0.12):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        bound = 1 / num_embeddings
        embedding = torch.zeros(num_embeddings, embedding_dim)
        embedding.uniform_(-bound, bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.jitter = Jitter(jitter) if jitter > 0 else None

    def forward(self, x):
        M, D = self.embedding.size()
        # unpack packed_sequence
        x, l = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        if self.jitter is not None:
            quantized = self.jitter(quantized)
        quantized = nn.utils.rnn.pack_padded_sequence(quantized, l, batch_first=True, enforce_sorted=False)
        return dict(quantized=quantized,
                    codes=indices.reshape(x.shape[0], x.shape[1]),
                    one_hot=encodings.reshape(x.shape[0], x.shape[1], -1),
                    loss=loss, perplexity=perplexity)
