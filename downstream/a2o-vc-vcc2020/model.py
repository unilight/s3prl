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
from torch.nn.functional import dropout

class RNNLayer(nn.Module):
    ''' RNN wrapper, includes time-downsampling'''

    def __init__(self, input_dim, module, bidirection, dim, dropout, layer_norm, sample_rate, proj, use_cell):
        super(RNNLayer, self).__init__()
        # Setup
        rnn_out_dim = 2 * dim if bidirection else dim
        self.out_dim = rnn_out_dim
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.sample_rate = sample_rate
        self.proj = proj
        self.use_cell = use_cell

        # Recurrent layer or cell
        if not use_cell:
            self.layer = getattr(nn, module.upper())(
                input_dim, dim, bidirectional=bidirection, num_layers=1, batch_first=True)
        else:
            self.cell = getattr(nn, module.upper()+"Cell")(input_dim, dim)

        # Regularizations
        if self.layer_norm:
            self.ln = nn.LayerNorm(rnn_out_dim)
        if self.dropout > 0:
            self.dp = nn.Dropout(p=dropout)

        # Additional projection layer
        if self.proj:
            self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)

    def forward(self, input_x, x_len):
        assert not self.use_cell

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

    def forward_one_step(self, input_x, z, c):
        assert self.use_cell

        new_z, new_c = self.cell(input_x, (z, c))

        # Normalizations
        if self.layer_norm:
            new_z = self.ln(new_z)
        if self.dropout > 0:
            new_z = self.dp(new_z)

        if self.proj:
            new_z = torch.tanh(self.pj(new_z))

        return new_z, new_c


class Model(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 resample_ratio,
                 stats,
                 ar,
                 hidden_dim,
                 lstmp_layers,
                 lstmp_dropout_rate,
                 lstmp_proj_dim,
                 lstmp_layernorm,
                 dropout_rate,
                 **kwargs):
        super(Model, self).__init__()

        self.ar = ar
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.resample_ratio = resample_ratio

        self.register_buffer("target_mean", torch.from_numpy(stats.mean_).float())
        self.register_buffer("target_scale", torch.from_numpy(stats.scale_).float())

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )

        self.lstmps = nn.ModuleList()
        for i in range(lstmp_layers):
            if ar and i == 0:
                rnn_input_dim = hidden_dim + output_dim
            else:
                rnn_input_dim = hidden_dim
            rnn_layer = RNNLayer(
                rnn_input_dim,
                "LSTM",
                False,
                hidden_dim,
                lstmp_dropout_rate,
                lstmp_layernorm,
                sample_rate=1,
                proj=True,
                use_cell=ar,
            )
            self.lstmps.append(rnn_layer)

        self.proj = torch.nn.Linear(hidden_dim, output_dim)

    def normalize(self, x):
        return (x - self.target_mean) / self.target_scale

    def forward(self, features, lens, targets = None):
        """Calculate forward propagation.
            Args:
            features: Batch of the sequences of input features (B, Lmax, idim).
            targets: Batch of the sequences of padded target features (B, Lmax, odim).
        """
        B = features.shape[0]
        
        # resample the input features according to resample_ratio
        features = features.permute(0, 2, 1)
        resampled_features = interpolate(features, scale_factor = self.resample_ratio)
        resampled_features = resampled_features.permute(0, 2, 1)
        lens = lens * self.resample_ratio

        # feed forward layer
        ffn_outputs = self.ffn(resampled_features) # (B, Lmax, hidden_dim)
        
        # LSTMP layers & projection
        if self.ar:
            if targets is not None:
                targets = targets.transpose(0, 1) # (Lmax, B, output_dim)
            predicted_list = []

            # initialize hidden states
            c_list = [ffn_outputs.new_zeros(B, self.hidden_dim)]
            z_list = [ffn_outputs.new_zeros(B, self.hidden_dim)]
            for _ in range(1, len(self.lstmps)):
                c_list += [ffn_outputs.new_zeros(B, self.hidden_dim)]
                z_list += [ffn_outputs.new_zeros(B, self.hidden_dim)]
            prev_out = ffn_outputs.new_zeros(B, self.output_dim)

            # step-by-step loop for autoregressive decoding
            for t, ffn_output in enumerate(ffn_outputs.transpose(0, 1)):
                concat = torch.cat([ffn_output, dropout(prev_out, 0.5)], dim=1) # each ffn_output has shape (B, hidden_dim)
                for i, lstmp in enumerate(self.lstmps):
                    lstmp_input = concat if i == 0 else z_list[i-1]
                    z_list[i], c_list[i] = lstmp.forward_one_step(lstmp_input, z_list[i], c_list[i])
                predicted_list += [self.proj(z_list[-1]).view(B, self.output_dim, -1)] # projection is done here to ensure output dim
                prev_out = targets[t] if targets is not None else predicted_list[-1].squeeze(-1) # targets not None = teacher-forcing
                prev_out = self.normalize(prev_out) # apply normalization
            predicted = torch.cat(predicted_list, dim=2)
            predicted = predicted.transpose(1, 2)  # (B, hidden_dim, Lmax) -> (B, Lmax, hidden_dim)
        else:
            predicted = ffn_outputs
            for i, lstmp in enumerate(self.lstmps):
                predicted, lens = lstmp(predicted, lens)
        
            # projection layer
            predicted = self.proj(predicted)

        return predicted, lens
