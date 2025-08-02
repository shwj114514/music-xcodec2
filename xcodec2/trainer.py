
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW
from itertools import chain

from xcodec2.common.schedulers import WarmupLR
from xcodec2.criterions import GANLoss, MultiResolutionMelSpectrogramLoss, MultiResolutionSTFTLoss
from xcodec2.module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
import typing as tp
import torch.nn as nn

from xcodec2.model import XCodec2Model

class XCodec2Trainer(pl.LightningModule):
    def __init__(
        self, 
        model:XCodec2Model,
        mpd_config,
        mstft_config,
        train_config,
        sample_rate=16000,
        use_feat = False
    ):
        super().__init__()
        self.model = model
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=mpd_config["periods"],
            max_downsample_channels=mpd_config["max_downsample_channels"],
            channels=mpd_config["channels"],
            channel_increasing_factor=mpd_config["channel_increasing_factor"],
        )
        self.spec_d = SpecDiscriminator(
            stft_params=mstft_config["stft_params"],
            in_channels=mstft_config["in_channels"],
            out_channels=mstft_config["out_channels"],
            kernel_sizes=mstft_config["kernel_sizes"],
            channels=mstft_config["channels"],
            max_downsample_channels=mstft_config["max_downsample_channels"],
            downsample_scales=mstft_config["downsample_scales"],
            use_weight_norm=mstft_config["use_weight_norm"],
        )

        self.train_config = train_config
        self.criteria = nn.ModuleDict()
        if train_config["use_mel_loss"]:
            self.criteria["mel_loss"] = MultiResolutionMelSpectrogramLoss(sample_rate=sample_rate)
        if train_config["use_stft_loss"]:
            self.criteria["stft_loss"] = MultiResolutionSTFTLoss(
                fft_sizes=train_config["stft_loss_params"]["fft_sizes"],
                hop_sizes=train_config["stft_loss_params"]["hop_sizes"],
                win_sizes=train_config["stft_loss_params"]["win_lengths"]
            )
        if train_config["use_feat_match_loss"]:
            self.criteria["fm_loss"] = torch.nn.L1Loss()

        self.criteria["gan_loss"] = GANLoss()
        self.criteria["l1_loss"] = torch.nn.L1Loss()
        self.criteria["mse_loss"] = torch.nn.MSELoss()
        print(self.criteria)

        self.automatic_optimization = False
        self.use_feat = use_feat
        print(f"use_feat = {self.use_feat}")

    def forward(self, wav):

        return self.model(wav)

    def compute_disc_loss(self, gt_wav, gen_wav):
        D_real = self.mpd(gt_wav)
        D_fake = self.mpd(gen_wav.detach())

        real_losses = []
        fake_losses = []
        gan_loss = self.criteria["gan_loss"]
        for i in range(len(D_real)):
            real_loss_i, fake_loss_i = gan_loss.disc_loss(D_real[i][-1], D_fake[i][-1])
            real_losses.append(real_loss_i)
            fake_losses.append(fake_loss_i)

        Dspec_real = self.spec_d(gt_wav)
        Dspec_fake = self.spec_d(gen_wav.detach())
        for i in range(len(Dspec_real)):
            real_loss_i, fake_loss_i = gan_loss.disc_loss(Dspec_real[i][-1], Dspec_fake[i][-1])
            real_losses.append(real_loss_i)
            fake_losses.append(fake_loss_i)

        real_loss = sum(real_losses)
        fake_loss = sum(fake_losses)
        disc_loss = real_loss + fake_loss
        lambda_disc = self.train_config["lambdas"]["lambda_disc"]
        disc_loss = lambda_disc * disc_loss

        return disc_loss, real_loss, fake_loss

    def compute_gen_loss(
        self, 
        gt_wav,
        gen_wav,
        vq_loss,
        semantic_target,
        semantic_recon
    ):

        cfg = self.train_config
        gen_loss_total = 0.0
        out_dict = {}

        # 1) MelLoss
        if cfg["use_mel_loss"]:
            mel_loss_val = self.criteria["mel_loss"](gen_wav.squeeze(1), gt_wav.squeeze(1))
            gen_loss_total += cfg["lambdas"]["lambda_mel_loss"] * mel_loss_val
            out_dict["mel_loss"] = mel_loss_val


        # 2) GAN Loss
        gan_loss = self.criteria["gan_loss"]
        D_fake = self.mpd(gen_wav)
        adv_loss_list = [gan_loss.gen_loss(d[-1]) for d in D_fake]

        Dspec_fake = self.spec_d(gen_wav)
        adv_loss_list += [gan_loss.gen_loss(d[-1]) for d in Dspec_fake]

        adv_loss_val = sum(adv_loss_list)
        gen_loss_total += adv_loss_val * cfg["lambdas"]["lambda_adv"]
        out_dict["adv_loss"] = adv_loss_val

        # 3) Feature Matching
        if cfg["use_feat_match_loss"]:
            fm_loss = 0.0
            with torch.no_grad():
                D_real = self.mpd(gt_wav)
            for i in range(len(D_fake)):
                for j in range(len(D_fake[i]) - 1):
                    fm_loss += F.l1_loss(D_fake[i][j], D_real[i][j].detach())
            
            with torch.no_grad():
                Dspec_real = self.spec_d(gt_wav)
            spec_fm_loss = 0.0
            for i in range(len(Dspec_fake)):
                for j in range(len(Dspec_fake[i]) - 1):
                    spec_fm_loss += F.l1_loss(Dspec_fake[i][j], Dspec_real[i][j].detach())

            fm_loss = fm_loss + spec_fm_loss
            gen_loss_total += fm_loss * cfg["lambdas"]["lambda_feat_match_loss"]
            out_dict["fm_loss"] = fm_loss

        # 4) VQ Loss
        if vq_loss is not None:
            vq_loss_val = sum(vq_loss)
            gen_loss_total += vq_loss_val
            out_dict["vq_loss"] = vq_loss_val

        # 5) semantic reconstruction loss
        semantic_recon_loss = F.mse_loss(semantic_recon, semantic_target)
        gen_loss_total += cfg["lambdas"]["lambda_semantic_loss"] * semantic_recon_loss
        out_dict["semantic_recon_loss"] = semantic_recon_loss

        out_dict["gen_loss"] = gen_loss_total
        return out_dict

    def training_step(self, batch, batch_idx):
        # wav  [4, 1, 240000]
        if self.use_feat:
            wav, feat,meta_data = batch
            out = self.model(wav,feat = feat)

        else:
            wav, meta_data = batch
            out = self.model(wav)



        gt_wav = out["gt_wav"]            # [B, 1, T]
        gen_wav = out["gen_wav"]          # [B, 1, T]
        vq_loss = out["vq_loss"]          # list of tensor
        semantic_target = out["semantic_target"]  # [B, 1024, T']
        semantic_recon = out["semantic_recon"]    # [B, 1024, T']
        gen_opt, disc_opt = self.optimizers()
        gen_sch, disc_sch = self.lr_schedulers()

        disc_loss, real_loss, fake_loss = self.compute_disc_loss(gt_wav, gen_wav)
        disc_opt.zero_grad()
        self.manual_backward(disc_loss)
        torch.nn.utils.clip_grad_norm_(disc_opt.param_groups[0]["params"], self.train_config["disc_grad_clip"])
        disc_opt.step()
        if disc_sch is not None:
            disc_sch.step()

        self.set_discriminator_grad(False)

        gen_dict = self.compute_gen_loss(
            gt_wav, gen_wav, vq_loss, semantic_target, semantic_recon
        )
        gen_loss = gen_dict["gen_loss"]

        gen_opt.zero_grad()
        self.manual_backward(gen_loss)
        torch.nn.utils.clip_grad_norm_(gen_opt.param_groups[0]["params"], self.train_config["gen_grad_clip"])
        gen_opt.step()
        if gen_sch is not None:
            gen_sch.step()

        self.set_discriminator_grad(True)

        self.log("train/disc_loss", disc_loss, prog_bar=True, on_step=True)
        self.log("train/real_loss", real_loss, prog_bar=False, on_step=True)
        self.log("train/fake_loss", fake_loss, prog_bar=False, on_step=True)
        for k, v in gen_dict.items():
            self.log(f"train/{k}", v, prog_bar=(k == "gen_loss"), on_step=True)
        
        return {"loss": gen_loss + disc_loss}

    def validation_step(self, batch, batch_idx):
        wav, meta_data = batch
        out = self.model(wav)
        l1 = F.l1_loss(out["gen_wav"], out["gt_wav"])
        self.log("val/l1_loss", l1, prog_bar=True)



    def configure_optimizers(self) -> tp.Tuple[tp.List[torch.optim.Optimizer], tp.List[torch.optim.lr_scheduler._LRScheduler]]:
        """
            https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html
        """
        gen_params = chain(self.model.parameters())
        disc_params = chain(
            self.mpd.parameters(),
            self.spec_d.parameters(),
        )

        gen_optim_params = self.train_config["gen_optim_params"]
        disc_optim_params = self.train_config["disc_optim_params"]

        gen_opt = AdamW(gen_params, **gen_optim_params)
        disc_opt = AdamW(disc_params, **disc_optim_params)

        gen_sch = WarmupLR(gen_opt, **self.train_config["gen_schedule_params"])
        disc_sch = WarmupLR(disc_opt, **self.train_config["disc_schedule_params"])

        return [gen_opt, disc_opt], [gen_sch, disc_sch]


    def set_discriminator_grad(self, flag=True):
        for p in self.mpd.parameters():
            p.requires_grad = flag
        for p in self.spec_d.parameters():
            p.requires_grad = flag


import os
import torch
from pytorch_lightning import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torchaudio
from pathlib import Path
class DemoCallback(Callback):
    def __init__(
        self, 
        demo_every=10000, 
        sample_rate=16000
    ):
        super().__init__()
        self.demo_every = demo_every
        self.sample_rate = sample_rate

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.demo_every != 0 or self.demo_every < 0:
            return

        pl_module.eval()
        if pl_module.use_feat:
            wav, feat,meta_data = batch
            out = pl_module.model(wav,feat = feat)

        else:
            wav, meta_data = batch
            out = pl_module.model(wav)

        print(f"generating.... >>>> trainer.global_step = {trainer.global_step}  sample_dir = {trainer.default_root_dir}")

        gen_wav = out.get("gen_wav", None)
        gen_wav = gen_wav.detach().cpu()
        gt_wav = out["gt_wav"]

        step_str = f"{trainer.global_step:08d}"
        sample_dir = os.path.join(trainer.default_root_dir, step_str)
        os.makedirs(sample_dir, exist_ok=True)

        for idx in range(gen_wav.shape[0]):
            recon_audio = gen_wav[idx]
            ori_audio = gt_wav[idx]

            stem = Path(meta_data[idx].get('file_name')).stem
            recon_save_path = os.path.join(sample_dir, f"recon_{stem}.wav")
            ori_save_path = os.path.join(sample_dir, f"ori_{stem}.wav")
            torchaudio.save(recon_save_path, recon_audio.detach().cpu(), sample_rate=self.sample_rate)
            torchaudio.save(ori_save_path, ori_audio.detach().cpu(), sample_rate=self.sample_rate)
        pl_module.train()




def create_training_wrapper_from_config(trainer_config, model):
    train_config = trainer_config['training']

    sample_rate = trainer_config['sample_rate']

    model_config = trainer_config['model']
    mpd_config = model_config["mpd"]
    mstft_config = model_config["mstft"]

    use_feat = trainer_config.get("use_feat",False)



    training_wrapper = XCodec2Trainer(
        model = model,
        mpd_config = mpd_config,
        mstft_config = mstft_config,
        train_config = train_config,
        sample_rate=sample_rate,
        use_feat = use_feat
    )

    return training_wrapper