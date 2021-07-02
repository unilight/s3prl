# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the LSTMP model ]
#   Reference    [ `WaveNet Vocoder with Limited Training Data for Voice Conversion`, Interspeech 2018 ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn.functional import interpolate

class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''

    def __init__(self, input_dim, module, bidirection, dim, dropout, layer_norm, sample_rate, proj):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2 * dim if bidirection else dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.proj = proj

        # Recurrent layer
        self.layer = getattr(nn, module.upper())(
            input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):
        # Forward RNN
        if not self.training:
            self.layer.flatten_parameters()

        input_x = pack_padded_sequence(input_x, x_len, batch_first=True, enforce_sorted=False)
        output, _ = self.layer(input_x)
        output, x_len = pad_packed_sequence(output, batch_first=True)

        # Normalizations
        if self.layer_norm:
            output = self.ln(output)
        if self.dropout > 0:
            output = self.dp(output)

        # Perform Downsampling
        if self.sample_rate > 1:
            output, x_len = downsample(output, x_len, self.sample_rate, 'drop')

        if self.proj:
            output = torch.tanh(self.pj(output))

        return output, x_len


class Model(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 resample_ratio,
                 hidden_dim,
                 lstmp_layers,
                 lstmp_dropout_rate,
                 lstmp_proj_dim,
                 lstmp_layernorm,
                 dropout_rate,
                 **kwargs):
        super(Model, self).__init__()

        self.resample_ratio = resample_ratio

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )

        self.lstmps = nn.ModuleList()
        for i in range(lstmp_layers):
            rnn_layer = RNNLayer(
                hidden_dim,
                "LSTM",
                False,
                hidden_dim,
                lstmp_dropout_rate,
                lstmp_layernorm,
                1,
                True,
            )
            self.lstmps.append(rnn_layer)

        self.proj = torch.nn.Linear(hidden_dim, output_dim)


    def forward(self, features, lens):
        
        # resample the input features according to resample_ratio
        features = features.permute(0, 2, 1)
        resampled_features = interpolate(features, scale_factor = self.resample_ratio)
        resampled_features = resampled_features.permute(0, 2, 1)
        lens = lens * self.resample_ratio

        # feed forward layer
        predicted = self.ffn(resampled_features)
        
        # LSTMP layers
        for i, lstmp in enumerate(self.lstmps):
            predicted, lens = lstmp(predicted, lens)
        
        # projection layer
        predicted = self.proj(predicted)

        return predicted, lens
