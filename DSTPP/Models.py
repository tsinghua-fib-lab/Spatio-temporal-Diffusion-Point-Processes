import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import DSTPP.Constants as Constants
from DSTPP.Layers import EncoderLayer


def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq, dim=2):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()[:2]
    subsequent_mask = torch.triu(
        torch.ones((dim, len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1).permute(1,2,0)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1,-1)  # b x ls x ls
    return subsequent_mask



class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,device, loc_dim):
        super().__init__()

        self.d_model = d_model
        self.loc_dim = loc_dim

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        # event loc embedding
        self.event_emb = nn.Sequential(
          nn.Linear(self.loc_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_temporal = nn.Modulelist([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask


    def forward(self, event_loc, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_loc)
        
        slf_attn_mask = slf_attn_mask[:,:,:,0]

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output



class Encoder_ST(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,device, loc_dim,CosSin = False):
        super().__init__()

        self.d_model = d_model
        self.loc_dim = loc_dim

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        # event loc embedding
        self.event_emb_temporal = nn.Sequential(
          nn.Linear(1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        self.event_emb_loc = nn.Sequential(
          nn.Linear(self.loc_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_loc = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_temporal = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        self.position_vec = self.position_vec.to(time)
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_loc, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        enc_output_temporal = self.temporal_enc(event_time, non_pad_mask)

        enc_output_loc = self.event_emb_loc(event_loc)

        enc_output = enc_output_temporal+enc_output_loc
        
        slf_attn_mask = slf_attn_mask[:,:,:,0]

        for index in range(len(self.layer_stack)):
            enc_output_loc, _ = self.layer_stack_loc[index](
                enc_output_loc,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            enc_output_temporal, _ = self.layer_stack_temporal[index](
                enc_output_temporal,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            enc_output, _ = self.layer_stack[index](
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        
        return enc_output, enc_output_temporal, enc_output_loc


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,device=None,loc_dim=2):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim = loc_dim
        )

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

    def forward(self, event_loc, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_loc: batch*seq_len*2;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim
        """

        non_pad_mask = get_non_pad_mask(event_time)
        
        enc_output = self.encoder(event_loc, event_time, non_pad_mask)
        enc_output = self.rnn(enc_output, non_pad_mask)

        return enc_output, non_pad_mask

class Transformer_ST(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,device=None,loc_dim=2,CosSin=False):
        super().__init__()

        self.encoder = Encoder_ST(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim = loc_dim,
            CosSin = CosSin
        )

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)
        self.rnn_temporal = RNN_layers(d_model, d_rnn)
        self.rnn_spatial = RNN_layers(d_model, d_rnn)

    def forward(self, event_loc, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_loc: batch*seq_len*2;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim
        """

        non_pad_mask = get_non_pad_mask(event_time)
        
        enc_output, enc_output_temporal, enc_output_loc = self.encoder(event_loc, event_time, non_pad_mask)

        assert (enc_output != enc_output_temporal).any() & (enc_output != enc_output_loc).any() & (enc_output_loc != enc_output_temporal).any()
        
        enc_output = self.rnn(enc_output, non_pad_mask)
        enc_output_temporal = self.rnn_temporal(enc_output_temporal, non_pad_mask)
        enc_output_loc = self.rnn_spatial(enc_output_loc, non_pad_mask)

        enc_output_all = torch.cat((enc_output_temporal, enc_output_loc, enc_output),dim=-1)

        return enc_output_all, non_pad_mask
