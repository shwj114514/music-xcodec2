import os
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from tqdm import tqdm

from argparse import ArgumentParser
from time import time
import torch.nn as nn

import torchaudio
from pathlib import Path
import typing as tp
import os

def get_filelist(
    folder: tp.Union[str, os.PathLike],
    extensions: tp.Optional[tp.List[str]] = None
) -> tp.List[str]:
    if extensions is None:
        extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    extensions = set(ext.lower() for ext in extensions)
    
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"The provided directory '{folder}' is not a valid directory.")
    filelist = [str(file) for file in path.rglob('*') if file.suffix.lower() in extensions]
    return filelist

if __name__ == '__main__':
    parser = ArgumentParser()

    TRAINER_CONFIG="config/trainer/muq_25hz.json"
    CKPT_PATH="pretrained/muq_25hz.pth"
    INPUT_FOLDER="test_audio"
    OUTPUT_FOLDER="exp_infer/muq_25hz"

    parser.add_argument('--input-dir', type=str, default=INPUT_FOLDER)
    parser.add_argument('--ckpt', type=str, default=CKPT_PATH)
    parser.add_argument('--output-dir',   type=str, default=OUTPUT_FOLDER)
    parser.add_argument("--trainer-config", type=str, default = TRAINER_CONFIG)
    

    
    args = parser.parse_args()
    DEVICE = "cuda:6"

    import json
    with open(args.trainer_config) as f:
        trainer_config: dict = json.load(f)
    from xcodec2.model import create_model_from_config
    from xcodec2.trainer import create_training_wrapper_from_config
    model_config = trainer_config["model"]
    MODEL_SR = trainer_config["sample_rate"]

    model = create_model_from_config(model_config)
    all_codes = []



    print(f'Load codec ckpt from {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    old_state_dict=ckpt['state_dict']
    # ckpt.keys() model.state_dict().keys()

    # mathc for lightning ckpt
    new_state_dict = {}
    for k, v in old_state_dict.items():
        if k.startswith("model."):
            new_key = k[len("model."):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)


    print(f"unexpected_keys = {unexpected_keys}")
    print(f"missing_keys = {missing_keys}")
    print(f"len(missing_keys) = {len(missing_keys) },len(unexpected_keys) = {len(unexpected_keys) } \n ")

    
    FMA_JSONL = "config/data/test_audio.jsonl"
    CNT = 500

    filelist = []
    with open(FMA_JSONL, 'r', encoding='utf-8') as f:
        for idx,line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if idx>CNT:
                break
            data = json.loads(line)
            filelist.append(data["path"])

    # filelist = get_filelist(args.input_dir)
    print(f'Found {len(filelist)} wavs in {args.input_dir}')
    
    model = model.to(DEVICE)
    st = time()
    for wav_path in tqdm(filelist):
        wav,sr = torchaudio.load(wav_path)
        wav = wav.to(DEVICE)

        if wav.shape[0] > 1:
            wav = wav.mean(0)
            wav = wav.unsqueeze(0)

        if sr != MODEL_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=MODEL_SR).to(DEVICE)
            wav = resampler(wav)
            sr = MODEL_SR
        wav = wav[:,:MODEL_SR*30]

        with torch.no_grad():
            output = model.forward(wav)
        gt_wav = output["gt_wav"]
        gen_wav = output["gen_wav"]
        vq_code = output["vq_code"]
        vq_code_flat = vq_code.flatten().cpu().numpy()
        all_codes.append(vq_code_flat)
        

    et = time()
    print(f'Inference ends, time: {(et-st)/60:.2f} mins')

    all_codes_array = np.concatenate(all_codes, axis=0)


    unique_codes, counts = np.unique(all_codes_array, return_counts=True)
    max_token = all_codes_array.max()
    total_count = len(all_codes_array)

    print(f"Total number of tokens: {total_count}")
    print(f"Max token index: {max_token}")
    print(f"Number of unique codes: {len(unique_codes)}")
    print(f"Codebook usage rate: {len(unique_codes) / max_token}")


    # for code, count in zip(unique_codes, counts):
    #     ratio = count / total_count
    #     print(f"code={code}, count={count}, ratio={ratio:.6f}")
