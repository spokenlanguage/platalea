import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from platalea.attention import BahdanauAttention
import platalea.dataset as D
import platalea.hardware


class TextDecoder(nn.Module):
    '''
    Attention decoder, following Bahdanau et al. (2015)
    [https://arxiv.org/abs/1409.0473]
    Borrowed from https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb
    '''
    def __init__(self, config):
        super(TextDecoder, self).__init__()
        emb = config['emb']
        drop = config['drop']
        att = config['att']
        rnn = config['rnn']
        out = config['out']

        # Define parameters
        self.max_output_length = config['max_output_length']
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 1.0)
        self.use_cuda = config.get('use_cuda', True)
        self.num_tokens = config['emb']['num_embeddings']
        self.rnn_layer_type = config['rnn_layer_type']

        # Define layers
        self.emb = nn.Embedding(**emb)
        self.drop = nn.Dropout(**drop)
        self.attn = BahdanauAttention(**att)
        self.RNN = self.rnn_layer_type(batch_first=True, **rnn)
        self.out = nn.Linear(**out)

    def init_state(self, tensor):
        state = tensor.new_zeros(1, tensor.shape[0], self.RNN.hidden_size)
        if type(self.RNN) == nn.LSTM:
            cell = tensor.new_zeros(1, tensor.shape[0], self.RNN.hidden_size)
            state = (state, cell)
        return state

    def hidden_state(self, state):
        if type(self.RNN) == nn.LSTM:
            return state[0]
        else:
            return state

    def forward(self, input, last_state, encoder_outputs):
        # Note that we will only be running forward for a single decoder time
        # step, but will use all encoder outputs

        # Get the embedding of the current input word (last output word)
        word_embedded = self.emb(input)
        word_embedded = self.drop(word_embedded)

        # Calculate attention weights and apply to encoder outputs
        lh = self.hidden_state(last_state).permute(1, 0, 2)  # SxBxN -> BxSxN
        attn_weights = self.attn(lh, encoder_outputs)
        context = attn_weights * encoder_outputs
        context = context.sum(dim=1)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context[:, None, :]), 2)
        output, state = self.RNN(rnn_input, last_state)

        # Final output layer
        output = self.out(torch.cat((output, context[:, None, :]), 2))
        output = F.log_softmax(output, dim=2)

        # Return final output, state, and attention weights (for
        # visualization)
        return output, state, attn_weights

    def decode(self, encoder_outputs, input_seq=None):
        # Prepare variables
        batch_size = encoder_outputs.shape[0]
        input = encoder_outputs.new_full((batch_size, 1),
                                         D.get_token_id(D.special_tokens.sos),
                                         dtype=torch.long)
        state = self.init_state(encoder_outputs)
        preds = None
        input_size = encoder_outputs.shape[1]
        attn_weights = torch.empty([batch_size, input_size, 0])
        if input_seq is not None:
            target_length = input_seq.shape[1]
        else:
            target_length = self.max_output_length
            # If input_seq is None (prediction mode), can end when all
            # sentences ended
            not_ended = encoder_outputs.new_full((batch_size, 1), True,
                                                 dtype=torch.bool)
        for di in range(target_length):
            output, state, att = self.forward(input, state, encoder_outputs)
            if preds is None:
                preds = output
            else:
                preds = torch.cat((preds, output), 1)
            attn_weights = torch.cat((attn_weights, att.detach().cpu()), 2)
            if input_seq is None:
                # Check for ended sequences
                no_eos = (output.argmax(dim=2) != D.get_token_id(D.special_tokens.eos))
                not_ended = not_ended & no_eos
                if not not_ended.any():
                    break
            # Select next input
            use_teacher_forcing = False
            if input_seq is not None:
                use_teacher_forcing = random.random() < self.teacher_forcing_ratio
            if use_teacher_forcing:
                # Teacher forcing: Use the ground-truth target as the next
                # input
                input = input_seq[:, di].view(-1, 1)
            else:
                # Without teacher forcing: use network's own prediction as
                # the next input
                input = output.argmax(dim=2)
        return preds, attn_weights

    def beam_search(self, encoder_outputs, beam_size):
        # Prepare variables
        predictions = np.empty([encoder_outputs.shape[0],
                                self.max_output_length], dtype=int)
        # Loop over sequences
        _device = platalea.hardware.device()
        for i_seq, eo in enumerate(encoder_outputs):
            eo = eo.unsqueeze(0)
            input = eo.new_full((1, 1), D.get_token_id(D.special_tokens.sos),
                                dtype=torch.long)
            state = self.init_state(eo)
            hyps = np.empty([1, 0], dtype=int)
            scores = np.zeros(1)
            best_ended_hyp = None
            # Loop over time steps
            for di in range(self.max_output_length):
                preds, st, _ = self.forward(input, state, eo)
                # Loop over hypotheses
                for idx_h, h in enumerate(hyps):
                    best_scores, best_ids = torch.topk(preds[idx_h], beam_size)
                    tmp_hyps = np.hstack([
                        np.repeat(h[np.newaxis, :], beam_size, axis=0),
                        best_ids.view(beam_size, 1).cpu()])
                    tmp_scores = scores[idx_h] + best_scores.squeeze().cpu()
                    if type(self.RNN) == nn.LSTM:
                        ht, ct = st
                        tmp_cell = ct[:, idx_h].repeat([beam_size, 1]).unsqueeze(0)
                    else:
                        ht = st
                    tmp_hidden = ht[:, idx_h].repeat([beam_size, 1]).unsqueeze(0)
                    if idx_h == 0:  # first iteration
                        new_hyps = tmp_hyps
                        new_scores = tmp_scores
                        new_hidden = tmp_hidden
                        if type(self.RNN) == nn.LSTM:
                            new_cell = tmp_cell
                    else:
                        new_hyps = np.vstack((new_hyps, tmp_hyps))
                        new_scores = np.hstack((new_scores, tmp_scores))
                        new_hidden = torch.cat((new_hidden, tmp_hidden), dim=1)
                        if type(self.RNN) == nn.LSTM:
                            new_cell = torch.cat((new_cell, tmp_cell), dim=1)
                        # Keep only <beam_size> best examples
                        new_order = np.argsort(-new_scores)
                        new_scores = new_scores[new_order][:beam_size]
                        new_hyps = new_hyps[new_order][:beam_size]
                        new_hidden = new_hidden[:, new_order][:, :beam_size]
                        if type(self.RNN) == nn.LSTM:
                            new_cell = new_cell[:, new_order][:, :beam_size]
                # Filter ended sequences
                ended = (new_hyps[:, -1] == D.get_token_id(D.special_tokens.eos)).nonzero()[0]
                best_ended_idx = None
                best_ended_score = -np.inf
                if len(ended) > 0:
                    if best_ended_hyp is None or new_scores[ended[0]] > best_ended_score:
                        best_ended_hyp = new_hyps[ended[0]]
                        best_ended_score = new_scores[ended[0]]
                        best_ended_idx = ended[0]
                if best_ended_idx is None:
                    if best_ended_hyp is not None:
                        best_ended_idx = np.searchsorted(-new_scores,
                                                         -best_ended_score)
                    else:
                        # keep all new hypothesis
                        best_ended_idx = beam_size
                if best_ended_idx == 0:
                    # already have best hypothesis, others will only get worse
                    break
                hyps = new_hyps[:best_ended_idx]
                scores = new_scores[:best_ended_idx]
                state = new_hidden[:, :best_ended_idx]
                if type(self.RNN) == nn.LSTM:
                    state = (state, new_cell[:, :best_ended_idx])
                # Select next input
                input = torch.unsqueeze(torch.from_numpy(hyps[:, -1]), 1).to(_device)
                # Duplicate encoder's output
                eo = eo[0].unsqueeze(0).repeat([hyps.shape[0], 1, 1])
            if best_ended_hyp is None:
                best_ended_hyp = new_hyps[0]
            predictions[i_seq, :len(best_ended_hyp)] = best_ended_hyp
        return predictions
