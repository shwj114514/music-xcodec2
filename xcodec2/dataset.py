import importlib
import io
import os
import posixpath
import random
import re
import subprocess
import time
import typing as tp
import json
from pathlib import Path
import typing as tp
import os

import numpy as np

import torch
from torchaudio import transforms as T
import torchaudio
from pedalboard.io import AudioFile

from .utils.audio_utils import is_silence
from .utils.torch_common import print_once
from .utils.data import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
from .utils.find import get_filelist

import librosa
import logging
logging.getLogger('librosa').setLevel(logging.ERROR)



class AudioFolderDataset(torch.utils.data.Dataset):
    # paths for all jsonls
    def __init__(
        self,
        filelist:tp.List[str],
        sample_size=65536,
        sample_rate=48000,
        relpath=None,
        random_crop=True,
        force_channels="mono",
        # augmentation
        augment_phase: bool = True
    ):
        assert force_channels in ['mono', 'stereo']

        super().__init__()
        self.relpath = relpath
        self.sr = sample_rate
        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.augs = torch.nn.Sequential(
            PhaseFlipper() if augment_phase else torch.nn.Identity()
        )

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)

        
        self.filelist = filelist
        print_once(f'->-> Found {len(self.filelist)} files.')

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        if ext == "mp3":
            with AudioFile(filename) as f:
                audio = f.read(f.frames)
                audio = torch.from_numpy(audio)
                in_sr = f.samplerate

        else:
            audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        file_path = self.filelist[idx]

        try:
            start_time = time.time()
            audio = self.load_file(file_path)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            # Encode the file to assist in prediction
            audio = self.encoding(audio)

            # Audio augmentations
            audio = self.augs(audio)
            audio = audio.clamp(-1, 1)

            info = {"path": file_path}
            if self.relpath:
                info["relpath"] = os.path.relpath(file_path, self.relpath)

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask
            info["load_time"] = time.time() - start_time


            # info.setdefault('prompt', 'This is a dummy prompt')
            return (audio, info)
        
        except Exception as e:
            print(f"e = {e}")
            # file = "29979930.mp3"
            # wav, sr = torchaudio.load(file)
            # wav = self.process_wav(wav,sr)
            return self[random.randrange(len(self))]

        return wav

def collation_fn(samples):
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], (int, float)):
            b = np.array(b)
        elif isinstance(b[0], torch.Tensor):
            b = torch.stack(b)
        elif isinstance(b[0], np.ndarray):
            b = np.array(b)
        else:
            b = b
        result.append(b)
    return result


def create_dataloader_from_config(
    dataset_config,
    batch_size: int,
    sample_size: int,
    sample_rate: int,
    audio_channels: int,
    num_workers: int = 4
):
    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type, "Dataset type must be specified in dataset config"
    assert audio_channels in [1, 2], f"Audio channel must be 1 or 2 -> found {audio_channels}."

    force_channels = "mono" if audio_channels == 1 else "stereo"

    if dataset_type == "audio_dir":
        audio_dir_configs = dataset_config.get("datasets", None)

        filelist = []
        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            if audio_dir_path.endswith(".jsonl"):
                sub_filelist = []
                with open(audio_dir_path, 'r') as f:
                    for line in f:
                        sub_filelist.append(json.loads(line.strip())['path'])
                filelist.extend(sub_filelist)    
                print(f"id {audio_dir_config['id']} have {len(sub_filelist)} files")

        train_set = AudioFolderDataset(
            filelist = filelist,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            force_channels=force_channels,
            relpath=None
        )

        return torch.utils.data.DataLoader(
            train_set,
            batch_size, 
            shuffle=True,
            num_workers=num_workers, 
            persistent_workers=True, 
            pin_memory=True, 
            drop_last=True, 
            collate_fn=collation_fn
        )

    
    else:
        import pdb;pdb.set_trace()

