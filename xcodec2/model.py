import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import soundfile as sf
import typing as tp
from xcodec2.vq import CodecEncoder,CodecDecoderVocos
from xcodec2.ssl_model import SSLModel

class XCodec2Model(nn.Module):
    def __init__(
        self,
        encoder:CodecEncoder,
        generator:CodecDecoderVocos,
        ssl_model:SSLModel,
        semantic_dim = 768,
        code_dim = 1024,
        decode_dim = 1024
    ):

        super().__init__()

        self.CodecEnc = encoder
        self.generator = generator

        vq_dim = self.generator.vq_dim
        hidden_dim = self.generator.hidden_dim
    
        # self.fc_prior = nn.Linear(
        #     1024 + 1024, 
        #     vq_dim
        # )

        # fusion layer
        self.fc_prior = nn.Linear(
            code_dim + encoder.hidden_dim, 
            vq_dim
        )

        self.fc_post_a = nn.Linear(vq_dim, hidden_dim)

        self.fc_post_s = nn.Linear(vq_dim, hidden_dim)


        from xcodec2.vq.module import SemanticDecoder, SemanticEncoder
        self.SemanticDecoder_module = SemanticDecoder(code_dim, semantic_dim, decode_dim)
        # self.SemanticDecoder_module = SemanticDecoder(code_dim, semantic_dim, semantic_dim)

        self.SemanticEncoder_module = SemanticEncoder(semantic_dim, code_dim, decode_dim)

        self.semantic_model = ssl_model

        # self.semantic_model.eval()
        # self.semantic_model.requires_grad_(False)



    def forward(self, wav,feat = None):
        if len(wav.shape) == 3 and wav.shape[1] == 1:
            wav = wav.squeeze(1)  # [B, T]

        # wav_pad = F.pad(wav, (160, 160))
        # feat = self.feature_extractor(wav_pad, sampling_rate=16000, return_tensors="pt")["input_values"]

        vq_emb = self.CodecEnc(wav.unsqueeze(1))
        vq_emb = vq_emb.transpose(1, 2)  # [B, 1024, T']

        # with torch.no_grad():

        #     semantic_target = self.semantic_model(wav,use_feat = use_feat)
        #     semantic_target = semantic_target.detach()

        #     for idx in range(semantic_target.size(0)):
        #         if torch.isnan(semantic_target[idx]).any():
        #             print(f"Warning: NaN detected in semantic_target at index {idx}!")
        #             # 保存对应的输入 wav 片段
        #             self.save_nan_wav(wav[idx], f"semantic_target_{idx}")

        with torch.no_grad():

            if feat is None:
                semantic_target = self.semantic_model(wav)
            else:
                semantic_target = self.semantic_model(feat)
            semantic_target.detach()

        semantic_target = semantic_target.transpose(1, 2)  # [B, 1024, T_sem]

        ''' for 10s audio
            muq 25hz: 
                semantic_target [1, 1024, 250] 
                vq_emb [1, 1024, 250]

            beats 50hz:
                semantic_target [1, 768, 448]
                vq_emb [1, 1024, 500]
        '''
        if semantic_target.shape[-1] != vq_emb.shape[-1]:
            semantic_target = F.interpolate(
                semantic_target,
                size=vq_emb.shape[-1],
                mode="linear",
                align_corners=False
            )

        semantic_target_processed = self.SemanticEncoder_module(semantic_target)
        

        vq_emb_concat = torch.cat([semantic_target_processed, vq_emb], dim=1)  # [B, 1024+1024, T']

        vq_emb_concat = self.fc_prior(vq_emb_concat.transpose(1, 2)).transpose(1, 2)  # [B, vq_dim, T']

        vq_post_emb, vq_code, vq_loss = self.generator(vq_emb_concat, vq=True)

        semantic_recon = self.fc_post_s(vq_post_emb.transpose(1, 2)).transpose(1, 2)
        semantic_recon = self.SemanticDecoder_module(semantic_recon)

        y_, _ = self.generator(self.fc_post_a(vq_post_emb.transpose(1, 2)), vq=False)
        y = wav.unsqueeze(1)
        return {
            "gt_wav": y, # [8, 1, 120000]
            "gen_wav": y_,
            "vq_loss": vq_loss,
            "vq_code": vq_code, # [8, 1, 125]
            "semantic_recon": semantic_recon,
            "semantic_target": semantic_target, # [8, 1024, 125]
        }

    def tokenize(self, wav,feat = None):
        if len(wav.shape) == 3 and wav.shape[1] == 1:
            wav = wav.squeeze(1)  # [B, T]

        vq_emb = self.CodecEnc(wav.unsqueeze(1))
        vq_emb = vq_emb.transpose(1, 2)  # [B, 1024, T']
        if feat is None:
            semantic_target = self.semantic_model(wav)
        else:
            semantic_target = self.semantic_model(feat)

        semantic_target = semantic_target.transpose(1, 2)  # [B, 1024, T_sem]
        semantic_target_processed = self.SemanticEncoder_module(semantic_target)
        
        
        if semantic_target_processed.shape[-1] != vq_emb.shape[-1]:
            semantic_target_processed = F.interpolate(
                semantic_target_processed,
                size=vq_emb.shape[-1],
                mode="linear",
                align_corners=False
            )

        vq_emb_concat = torch.cat([semantic_target_processed, vq_emb], dim=1)  # [B, 1024+1024, T']
        vq_emb_concat = self.fc_prior(vq_emb_concat.transpose(1, 2)).transpose(1, 2)  # [B, vq_dim, T']

        vq_post_emb, vq_code, vq_loss = self.generator(vq_emb_concat, vq=True)
        import pdb;pdb.set_trace()
        return vq_code
    
    def detokenize(self, token):
        if len(token.shape) == 2:
            token = token.unsqueeze(1)
        vq_post_emb = self.generator.quantizer.get_output_from_indices(token.transpose(1, 2))
        vq_post_emb = vq_post_emb.transpose(1, 2)
        # [4, 1024, 748]
        vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1,2)).transpose(1,2)
        # recon = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0].squeeze().detach().cpu().numpy()
        recon = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0]

        return recon

    def get_quantized_latent(self, wav,feat = None):
        # [6, 718890] ->[6, 1, 748] -> [6, 748, 2048]
        # torch.unique(vq_code).numel()
        vq_code = self.tokenize(wav = wav)
        vq_post_emb = self.generator.quantizer.get_output_from_indices(vq_code.transpose(1, 2))
        return vq_post_emb

    def decode_quantized_latent(self, latent):
        # vq_post_emb = latent.transpose(1, 2)
        vq_post_emb = latent

        # [4, 1024, 748]
        vq_post_emb = self.fc_post_a(vq_post_emb.transpose(1,2)).transpose(1,2)
        recon = self.generator(vq_post_emb.transpose(1, 2), vq=False)[0]

        # import torchaudio
        # torchaudio.save("test1.wav",recon[2].cpu(),sample_rate = 24000)
        return recon

    @torch.inference_mode()
    def inference(self, wav):
        vq_emb = self.CodecEnc(wav.unsqueeze(1))
        vq_post_emb, vq_code, vq_loss = self.generator(vq_emb, vq=True)
        y_, _ = self.generator(vq_post_emb, vq=False)
        return y_.squeeze(1)

    def save_nan_wav(self, wav, label):
        filename = os.path.join(self.ocwd, f"nan_{label}_{random.randint(1000,9999)}.wav")
        sr = self.cfg.preprocess.audio.sr
        wav_np = wav.cpu().numpy()
        sf.write(filename, wav_np, sr)
        print(f"Saved NaN wav snippet to {filename}")


def build_codec_encoder(encoder_config: tp.Dict):
    return CodecEncoder(
        ngf=encoder_config["ngf"],
        up_ratios=encoder_config["up_ratios"],
        dilations=encoder_config["dilations"],
        hidden_dim=encoder_config["hidden_dim"],
        depth=encoder_config["depth"],
        heads=encoder_config["heads"],
        pos_meb_dim=encoder_config["pos_meb_dim"],
    )

def build_codec_decoder(decoder_config: tp.Dict):
    return CodecDecoderVocos(
        hidden_dim=decoder_config["hidden_dim"],
        depth=decoder_config["depth"],
        heads=decoder_config["heads"],
        pos_meb_dim=decoder_config["pos_meb_dim"],
        hop_length=decoder_config.get("hop_length", 960),  # 默认值 960
        vq_num_quantizers=decoder_config["vq_num_quantizers"],
        vq_dim=decoder_config["vq_dim"],
        vq_commit_weight=decoder_config["vq_commit_weight"],
        vq_weight_init=decoder_config["vq_weight_init"],
        vq_full_commit_loss=decoder_config["vq_full_commit_loss"],
        codebook_size=decoder_config["codebook_size"],
        codebook_dim=decoder_config["codebook_dim"],
    )

def build_mpd(mpd_config: tp.Dict):
    from xcodec2.module import HiFiGANMultiPeriodDiscriminator
    return HiFiGANMultiPeriodDiscriminator(
        periods=mpd_config["periods"],
        max_downsample_channels=mpd_config["max_downsample_channels"],
        channels=mpd_config["channels"],
        channel_increasing_factor=mpd_config["channel_increasing_factor"],
    )

def build_spec_discriminator(mstft_config: tp.Dict):
    from xcodec2.module import SpecDiscriminator
    return SpecDiscriminator(
        stft_params=mstft_config["stft_params"],
        in_channels=mstft_config["in_channels"],
        out_channels=mstft_config["out_channels"],
        kernel_sizes=mstft_config["kernel_sizes"],
        channels=mstft_config["channels"],
        max_downsample_channels=mstft_config["max_downsample_channels"],
        downsample_scales=mstft_config["downsample_scales"],
        use_weight_norm=mstft_config["use_weight_norm"],
    )




def create_model_from_config(model_config) -> XCodec2Model:
    # model_type = model_config['model_type']
    codec_conifg = model_config["codec"]
    encoder_config = codec_conifg["encoder"]
    decoder_config = codec_conifg["decoder"]
    encoder = build_codec_encoder(encoder_config)
    decoder = build_codec_decoder(decoder_config)
    
    # ssl_mdoel = build_ssl_model()
    # semantic_name_or_path = model_config["semantic_model_or_path"]

    ssl_config = codec_conifg["ssl"]
    ssl_type = ssl_config.pop("type")
    if ssl_type == "hubert":
        from .ssl_model import HubertWrapper
        ssl_model = HubertWrapper(**ssl_config)
    elif ssl_type == "MuQ":
        from .ssl_model import MuQWrapper
        ssl_model = MuQWrapper(**ssl_config)
    elif ssl_type == "wav2vec2bert":
        from .ssl_model import Wav2Vec2BertWrapper
        ssl_model = Wav2Vec2BertWrapper(**ssl_config)
    elif ssl_type == "beats":
        from .ssl_model import BeatWrapper
        ssl_model = BeatWrapper(**ssl_config)
        
    else:
        import pdb;pdb.set_trace()
    xcodec2 = XCodec2Model(
        encoder = encoder,
        generator = decoder,
        ssl_model = ssl_model,
        semantic_dim = ssl_model.ssl_dim,
        code_dim = 1024,
        decode_dim = 1024
    )
    return xcodec2

def create_loaded_model_from_config(config_path,ckpt_path,only_load_encoder = False) -> XCodec2Model:
    import json
    with open(config_path) as f:
        xcode2_trainer_config: dict = json.load(f)
    model = create_model_from_config(xcode2_trainer_config["model"])


    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    old_state_dict=ckpt['state_dict']
    new_state_dict = {}
    for k, v in old_state_dict.items():
        if k.startswith("model."):
            new_key = k[len("model."):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    filtered_state_dict = {}

    for k, v in new_state_dict.items():
        if "resampler" in k:         
            print(f"resampler in {k}")
        else:
            filtered_state_dict[k] = v

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    print(f"$loaded xocdec2$: len(missing_keys) = {len(missing_keys) },missing_keys = {missing_keys} len(unexpected_keys) = {len(unexpected_keys) } \n ")
    
    if only_load_encoder:
        for name in list(model.generator._modules.keys()):
            if name != "quantizer": 
                del model.generator._modules[name]
                print(f"del name = {name}")
    return model