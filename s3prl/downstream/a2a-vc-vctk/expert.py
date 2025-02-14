# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the any-to-any voice conversion downstream wrapper ]
#   Author       [ Wen-Chin Huang (https://github.com/unilight) ]
#   Copyright    [ Copyright(c), Toda Lab, Nagoya University, Japan ]
"""*********************************************************************************************"""


import os
import numpy as np
from scipy.io.wavfile import write
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence

from .model import Model
from .dataset import VCTK_VCC2020Dataset
from .utils import make_non_pad_mask
from .utils import read_hdf5, write_hdf5
from .utils import logmelspc_to_linearspc, griffin_lim

FS = 16000

class Loss(nn.Module):
    """
    L1 loss module supporting (1) loss calculation in the normalized target feature space
                              (2) masked loss calculation
    """
    def __init__(self, stats):
        super(Loss, self).__init__()
        self.objective = torch.nn.L1Loss(reduction="mean")
        self.register_buffer("target_mean", torch.from_numpy(stats.mean_).float())
        self.register_buffer("target_scale", torch.from_numpy(stats.scale_).float())

    def normalize(self, x):
        return (x - self.target_mean) / self.target_scale

    def forward(self, x, y, x_lens, y_lens, device):
        # match the input feature length to acoustic feature length to calculate the loss
        if x.shape[1] > y.shape[1]:
            x = x[:, :y.shape[1]]
            masks = make_non_pad_mask(y_lens).unsqueeze(-1).to(device)
        if x.shape[1] <= y.shape[1]:
            y = y[:, :x.shape[1]]
            masks = make_non_pad_mask(x_lens).unsqueeze(-1).to(device)
        
        # calculate masked loss
        x_normalized = self.normalize(x)
        y_normalized = self.normalize(y.to(device))
        x_masked = x_normalized.masked_select(masks)
        y_masked = y_normalized.masked_select(masks)
        loss = self.objective(x_masked, y_masked)
        return loss

class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        
        # basic settings
        self.expdir = expdir
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.acoustic_feature_dim = self.datarc["fbank_config"]["n_mels"]
        self.fs = self.datarc["fbank_config"]["fs"]
        self.resample_ratio = self.fs / self.datarc["fbank_config"]["n_shift"] * upstream_rate / FS
        print('[Downstream] - resample_ratio: ' + str(self.resample_ratio))

        # load datasets
        self.train_dataset = VCTK_VCC2020Dataset('train', **self.datarc)
        self.dev_dataset = VCTK_VCC2020Dataset('dev', **self.datarc)
        self.test_dataset = VCTK_VCC2020Dataset('test', **self.datarc)

        # load statistics file if exists, and calculate if not found
        scaler = StandardScaler()
        stats_root = self.datarc["stats_root"]
        if not os.path.exists(stats_root):
            os.makedirs(stats_root)
        stats_path = os.path.join(stats_root, "stats.h5")
        if os.path.exists(stats_path):
            print("[Stats] - reading stats from " + str(stats_path))
            scaler.mean_ = read_hdf5(stats_path, "mean")
            scaler.scale_ = read_hdf5(stats_path, "scale")
        else:
            print("[Stats] - " + str(stats_path) + " does not exist. Reading data...")
            lmspcs = np.concatenate(self.train_dataset.get_all_lmspcs(), axis=0)
            print("[Stats] - " + str(stats_path) + " does not exist. Calculating statistics...")
            scaler.fit(lmspcs)
            write_hdf5(stats_path, "mean", scaler.mean_.astype(np.float32))
            write_hdf5(stats_path, "scale", scaler.scale_.astype(np.float32))
            print("[Stats] - writing stats to " + str(stats_path))
        self.stats = scaler

        # define model and loss
        self.model = Model(
            input_dim = self.upstream_dim,
            output_dim = self.acoustic_feature_dim,
            resample_ratio = self.resample_ratio,
            stats = self.stats,
            **self.modelrc
        )
        self.objective = Loss(self.stats)


    # Interface
    def get_dataloader(self, split):
        if split == 'train':
            return self._get_train_dataloader(self.train_dataset)            
        elif split == 'dev':
            return self._get_eval_dataloader(self.dev_dataset)
        elif split == 'test':
            return self._get_eval_dataloader(self.test_dataset)


    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )


    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=dataset.collate_fn
        )


    # Interface
    def forward(self,
                split,
                input_features,
                wavs,
                acoustic_features,
                acoustic_features_padded,
                acoustic_feature_lengths,
                wav_paths,
                ref_spk_embs,
                ref_spk_names, 
                records,
                **kwargs):

        device = input_features[0].device

        # padding
        input_feature_lengths = torch.IntTensor([feature.shape[0] for feature in input_features])
        input_features = pad_sequence(input_features, batch_first=True).to(device=device)
        ref_spk_embs = ref_spk_embs.to(device)
        
        # forward model
        if split in ["dev", "test"]:
            predicted_features, predicted_feature_lengths = self.model(input_features, input_feature_lengths, ref_spk_embs)
            # save the unnormalized features for dev and test sets
            records["predicted_features"] += predicted_features.cpu().numpy().tolist()
            records["feature_lengths"] += predicted_feature_lengths.cpu().numpy().tolist()
            records["wav_paths"] += wav_paths
            records["ref_spk_names"] += ref_spk_names
            records["wavs"] += wavs
        else:
            predicted_features, predicted_feature_lengths = self.model(input_features, input_feature_lengths, ref_spk_embs, acoustic_features_padded.to(device))

        # loss calculation (masking and normalization are done inside)
        loss = self.objective(predicted_features,
                              acoustic_features_padded,
                              predicted_feature_lengths,
                              acoustic_feature_lengths,
                              device)
        records['loss'].append(loss.item())

        return loss


    # interface
    def log_records(self, split, records, logger, global_step, batch_ids, total_batch_num, **kwargs):
        loss = torch.FloatTensor(records['loss']).mean().item()

        if split in ["dev", "test"]:

            hdf5_save_dir = os.path.join(self.expdir, str(global_step), split, "hdf5")
            wav_save_dir = os.path.join(self.expdir, str(global_step), split, "wav")
            os.makedirs(hdf5_save_dir, exist_ok=True)
            os.makedirs(wav_save_dir, exist_ok=True)

            for i, (wav_path, ref_spk_name) in enumerate(tqdm(list(zip(records["wav_paths"], records["ref_spk_names"]), dynamic_ncols=True, desc="Saving files"))):
                length = int(records["feature_lengths"][i])
                fbank = np.array(records["predicted_features"][i])[:length]
                if split == "dev":
                    hdf5_save_path = os.path.join(hdf5_save_dir, "_".join(wav_path.split("/")[-2:]).replace(".wav", ".h5"))
                    wav_save_path = os.path.join(wav_save_dir, "_".join(wav_path.split("/")[-2:]))
                elif split == "test":
                    num_samples = os.path.basename(wav_path).replace(".wav", "").split("_")[-1]
                    os.makedirs(os.path.join(hdf5_save_dir, num_samples), exist_ok=True)
                    os.makedirs(os.path.join(wav_save_dir, num_samples), exist_ok=True)
                    hdf5_save_path = os.path.join(hdf5_save_dir, num_samples, "_".join([ref_spk_name] + wav_path.split("/")[-2:]).replace(".wav", ".h5"))
                    wav_save_path = os.path.join(wav_save_dir, num_samples, "_".join([ref_spk_name] + wav_path.split("/")[-2:]))

                # save generated features into hdf5 files
                write_hdf5(hdf5_save_path, "feats", fbank)

                # mel fbank -> linear spectrogram
                spc = logmelspc_to_linearspc(
                    fbank,
                    fs=self.datarc["fbank_config"]["fs"],
                    n_mels=self.datarc["fbank_config"]["n_mels"],
                    n_fft=self.datarc["fbank_config"]["n_fft"],
                    fmin=self.datarc["fbank_config"]["fmin"],
                    fmax=self.datarc["fbank_config"]["fmax"],
                )
                # apply griffin lim algorithm
                y = griffin_lim(
                    spc,
                    n_fft=self.datarc["fbank_config"]["n_fft"],
                    n_shift=self.datarc["fbank_config"]["n_shift"],
                    win_length=self.datarc["fbank_config"]["win_length"],
                    window=self.datarc["fbank_config"]["window"],
                    n_iters=self.datarc["fbank_config"]["gl_iters"],
                )
                # save generated waveform
                write(
                    wav_save_path,
                    self.datarc["fbank_config"]["fs"],
                    (y * np.iinfo(np.int16).max).astype(np.int16),
                )

        print(f'{split} loss: {loss:.6f}')

        save_names = []
        logger.add_scalar(
            f'example/{split}-{loss}',
            loss,
            global_step=global_step
        )
        return save_names
