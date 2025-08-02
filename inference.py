import os
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from glob import glob
from tqdm import tqdm
from os.path import basename, join, exists
from xcodec2.vq.codec_encoder import CodecEncoder
from xcodec2.vq.codec_decoder_vocos import CodecDecoderVocos
from argparse import ArgumentParser
from time import time
from transformers import  AutoModel
import torch.nn as nn
from xcodec2.vq.module import SemanticDecoder,SemanticEncoder
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import torchaudio

from pathlib import Path

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='test_audio/input_test')
    parser.add_argument('--ckpt', type=str, default='ckpt/epoch=4-step=1400000.ckpt')
    parser.add_argument('--output-dir',   type=str, default='test_audio/output_test')
    parser.add_argument("--trainer-config", type=str, default = "config/trainer/mert_50hz.json")
        
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
    try:
        old_state_dict=ckpt['state_dict']
    except Exception as e:
        old_state_dict=ckpt

    # ckpt.keys() model.state_dict().keys()

    # mathc for lightning ckpt
    new_state_dict = {}
    for k, v in old_state_dict.items():
        # 如果以 "model." 开头，则去掉这个前缀
        if k.startswith("model."):
            new_key = k[len("model."):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)


    print(f"unexpected_keys = {unexpected_keys}")
    print(f"missing_keys = {missing_keys}")
    print(f"len(missing_keys) = {len(missing_keys) },len(unexpected_keys) = {len(unexpected_keys) } \n ")
 

    SEC = 30

    wav_dir = args.output_dir
    os.makedirs(wav_dir, exist_ok=True)

    
    # wav_paths = glob(join(args.input_dir, '*.flac')) #
    # both wav and flac and mp3
    wav_paths = glob(os.path.join(args.input_dir, '**', '*.wav'), recursive=True)
    flac_paths = glob(os.path.join(args.input_dir, '**', '*.flac'), recursive=True)
    mp3_paths = glob(os.path.join(args.input_dir, '**', '*.mp3'), recursive=True)

    wav_paths = wav_paths + flac_paths + mp3_paths
    print(f'Found {len(wav_paths)} wavs in {args.input_dir}')
    
    model = model.to(DEVICE)
    st = time()
    for wav_path in tqdm(wav_paths):
        wav,sr = torchaudio.load(wav_path)
        wav = wav

        if wav.shape[0] > 1:
            wav = wav.mean(0)
            wav = wav.unsqueeze(0)

        if sr != MODEL_SR:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=MODEL_SR)
            wav = resampler(wav)
            sr = MODEL_SR
        wav = wav[:,:MODEL_SR*SEC]

        # if trainer_config["use_feat"] == True:
        if trainer_config.get("use_feat", False):
            feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
            resampler_16k = torchaudio.transforms.Resample(MODEL_SR, 16000)

        else:
            feat = None
        '''
         {
            "gt_wav": y,
            "gen_wav": y_,
            "vq_loss": vq_loss,
            "vq_code": vq_code,
            "semantic_recon": semantic_recon,
            "semantic_target": semantic_target,
        }
        
        '''

        with torch.no_grad():

            # output = model.forward(wav,use_feat = True)
            if trainer_config.get("use_feat", False):

                wav_16k = resampler_16k(wav)
                wav_pad = F.pad(wav_16k, (160, 160))
                feat = feature_extractor(wav_pad, sampling_rate=16000, return_tensors="pt") .data['input_features']
                # feat = feat.squeeze(0)  # [1, 250, 160] -> [250, 160]
                feat = feat.to(DEVICE)

            wav = wav.to(DEVICE)
            output = model.forward(wav,feat = feat)
        # import pdb;pdb.set_trace()
        # [1, 720000] ->  [1, 1, 750]

        gt_wav = output["gt_wav"]
        gen_wav = output["gen_wav"]
        vq_code = output["vq_code"]
        vq_code_flat = vq_code.flatten().cpu().numpy()
        all_codes.append(vq_code_flat)

        
        # stem = Path(wav_path).stem
        # gt_wav_save_path = os.path.join(wav_dir,f"{stem}_gt.wav")
        # gen_wav_save_path = os.path.join(wav_dir,f"{stem}_gen.wav")
        # torchaudio.save(gt_wav_save_path,gt_wav.cpu().squeeze(0),MODEL_SR)
        # torchaudio.save(gen_wav_save_path,gen_wav.cpu().squeeze(0),MODEL_SR)

        name = Path(wav_path).name
        gen_wav_save_path = os.path.join(wav_dir,name)
        torchaudio.save(gen_wav_save_path,gen_wav.cpu().squeeze(0),MODEL_SR)

    et = time()
    print(f'Inference ends, time: {(et-st)/60:.2f} mins')

    all_codes_array = np.concatenate(all_codes, axis=0)


    unique_codes, counts = np.unique(all_codes_array, return_counts=True)
    max_token = all_codes_array.max()
    total_count = len(all_codes_array)

    print(f"Total number of tokens: {total_count}")
    print(f"Max token length: {max_token}")
    print(f"Number of unique codes: {len(unique_codes)}")
    print(f"Saved in {wav_dir}")


    # for code, count in zip(unique_codes, counts):
    #     ratio = count / total_count
    #     print(f"code={code}, count={count}, ratio={ratio:.6f}")
