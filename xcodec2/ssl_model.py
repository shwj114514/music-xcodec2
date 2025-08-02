import torch
from torch import nn
import torch.nn.functional as F
import typing as tp

class SSLModel(nn.Module):
 
    def __init__(
            self,
            audio_channels: int = 1,
            enable_grad: bool = False,
            layer_idx: int = -1,
    ):
        super().__init__()

        self.audio_channels = audio_channels
        self.enable_grad = enable_grad
        self.layer_idx = layer_idx

    def interpolate(self, x, target_len):
        raise NotImplementedError

    def get_latent(self, x, layer_idx: int = -1):
        raise NotImplementedError


# model_name_or_path="ZhenYe234/hubert_base_general_audio" 50hz
# model_name_or_path="facebook/hubert-base-ls960"
# model_name_or_path="m-a-p/MERT-v1-330M"
class HubertWrapper(SSLModel):
    def __init__(
        self,
        model_name_or_path="m-a-p/MERT-v1-95M",
        enable_grad = False
    ):
        '''
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L31
        def __cal~l__(
            raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],

            Must be mono channel audio, not stereo,
        '''
        super().__init__(enable_grad=enable_grad, audio_channels=1)
        # from transformers import HubertModel,AutoProcessor
        from transformers import AutoModel, AutoProcessor, Wav2Vec2FeatureExtractor

        self.ssl_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        # self.processor = AutoProcessor.from_pretrained(model_name_or_path,trust_remote_code=True)
        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path, trust_remote_code=True)
        except Exception as e:
            print(f"load processor = {model_name_or_path} error")
            self.processor = None
        self.sample_rate = 24000
        self.ssl_dim = 768
        
        if self.enable_grad == False:
            self.ssl_model.eval().requires_grad_(False)

    def forward(self, wav : torch.Tensor,use_feat = False):
        wav_pad = F.pad(wav, (160, 160))
        # [1, 6, 160320]

        import torchaudio
        wav_16k = torchaudio.functional.resample(
            wav_pad, orig_freq=24000, new_freq=16000
        )
        if self.processor is not None:        
            feat = self.processor(wav_16k, sampling_rate=16000, return_tensors="pt")['input_values']
        else:
            feat = wav
        
        # semantic_target

        if use_feat:
            batch_features = []
            for single_wav in wav_pad:
                feat = self.processor(single_wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt")['input_values']
                feat = feat.to(single_wav.device)  # [1, seq_len]
                batch_features.append(feat)

            feat = torch.cat(batch_features, dim=0)  # [batch, seq_len]
        else:
            feat = wav_pad

        latent = self.ssl_model(feat, output_hidden_states=True).last_hidden_state
        return latent

class MuQWrapper(SSLModel):
    def __init__(
        self,
        model_name_or_path="OpenMuQ/MuQ-large-msd-iter",
        enable_grad: bool = False,
    ):
        super().__init__(enable_grad=enable_grad, audio_channels=1)
        # from transformers import HubertModel,AutoProcessor
        from muq import MuQ
        self.ssl_model =  MuQ.from_pretrained(model_name_or_path).eval()

        self.sample_rate = 24000
        self.ssl_dim = 1024

        if self.enable_grad == False:
            self.ssl_model.eval().requires_grad_(False)

    def forward(self, wav : torch.Tensor,use_feat = False):
        
        # [8, 125, 1024]  [8, 1, 120000]

        # [1, 720000] -> [1, 750, 1024]
        with torch.cuda.amp.autocast(enabled=False):
            latent = self.ssl_model(wav.unsqueeze(1), output_hidden_states=True).last_hidden_state #step2 NAN
        return latent

from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
import torchaudio
class Wav2Vec2BertWrapper(SSLModel):
    def __init__(
        self,
        model_name_or_path="facebook/w2v-bert-2.0",
        enable_grad: bool = False,
    ):
        super().__init__(enable_grad=enable_grad, audio_channels=1)
        # from transformers import HubertModel,AutoProcessor
        self.ssl_model = Wav2Vec2BertModel.from_pretrained(model_name_or_path, output_hidden_states=True)
        # self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_name_or_path)

        self.sample_rate = 16000
        self.ssl_dim = 1024

        if self.enable_grad == False:
            self.ssl_model.eval().requires_grad_(False)

    def forward(self, feat : tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]],use_feat = False):
 
        if use_feat:
            # mel input
            feat = feat
        else:
            # wav input
            # [1, 500, 160]  -> [B, 160320] -> [B, 500, 160] 
            wav_pad = F.pad(feat, (160, 160))
            feat = self.feature_extractor(wav_pad, sampling_rate=16000, return_tensors="pt") .data['input_features']    
        latent = self.ssl_model(feat, output_hidden_states=True).last_hidden_state

        return latent


class BeatWrapper(SSLModel):
    def __init__(
        self,
        model_name_or_path="pretrained/betas/BEATs_iter3_plus_AS2M.pt",
        enable_grad: bool = False,
        data_sr = 44100
    ):
        super().__init__(enable_grad=enable_grad, audio_channels=1)

        self.model_sr = 16000
        self.data_sr = data_sr
        print(f"self.data_sr = {self.data_sr } !!!!!!")
        self.ssl_dim = 768


        from beats.BEATs import BEATs, BEATsConfig
        from transformers import Wav2Vec2Config

        # load the pre-trained checkpoints
        checkpoint = torch.load(model_name_or_path)

        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        self.ssl_model = BEATs_model
        if self.enable_grad == False:
            self.ssl_model.eval().requires_grad_(False)

        """
            audio_input_16khz = torch.randn(1, 160000)
            padding_mask = torch.zeros(1, 160000).bool()
            representation = self.ssl_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
        

            padding_mask = torch.torch.zeros_like(wav).bool()
            representation = self.ssl_model.extract_features(audio_input_16khz, padding_mask=padding_mask)[0]
            representation = self.ssl_model.extract_features(audio_input_16khz)
        
        """

        if self.data_sr != self.model_sr:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.data_sr,
                new_freq=self.model_sr
            )


    # def get_latent(self, wav : torch.Tensor,use_feat = False,in_sr = 24000):
    def forward(self, wav : torch.Tensor,use_feat = False,in_sr = None):
        '''
            [1, 720000] -> [1, 480000] -> [1, 1496, 768]
            30s 1500token  1s 30token
        '''
        # [8, 479260]
        if in_sr == None:
            in_sr =  self.data_sr
        if in_sr == self.model_sr:
            wav = wav
        elif in_sr == self.data_sr:
            # wav = torchaudio.functional.resample(
            #     wav, orig_freq=in_sr, new_freq=self.sample_rate
            # )
            wav = self.resampler(wav)
        else:
            resampler = torchaudio.transforms.Resample(
                orig_freq=in_sr,
                new_freq=self.model_sr
            )
            wav = resampler(wav)

        '''
        [8, 248, 768] 50hz
        16000 / 50 =320
        [8, 80000]  5s

        1hz 

        pad到128维度  [8, 256, 768]
        '''
        # wav_pad = F.pad(wav, (1280, 1280))

        latent = self.ssl_model.extract_features(wav)[0]
        # latent = torch.zeros_like(latent).to(latent)
        return latent


if __name__ == "__main__":

    ssl_model = HubertWrapper(model_name_or_path="ZhenYe234/hubert_base_general_audio")    
    # model = MuQWrapper("OpenMuQ/MuQ-large-msd-iter")
    # dummy_input = torch.randn(3, 24000*5)


    ssl_model = Wav2Vec2BertWrapper()
    dummy_input_10s = torch.randn(4,10*ssl_model.sample_rate)

    with torch.no_grad():
        latent = ssl_model(dummy_input_10s)

    print("Input shape:", dummy_input_10s.shape)
    # [3, 375, 768]
    print("Output latent shape:", latent.shape)