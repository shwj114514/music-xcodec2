import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class STFTLoss(nn.Module):
    def __init__(self, fft_size: int, hop_size: int, win_size: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer('window', torch.hann_window(win_size))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x_stft = torch.stft(
            x, n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_size,
            window=self.window, return_complex=True
        )
        y_stft = torch.stft(
            y, n_fft=self.fft_size, hop_length=self.hop_size, win_length=self.win_size,
            window=self.window, return_complex=True
        )

        x_mag = torch.abs(x_stft)
        y_mag = torch.abs(y_stft)

        # Spectral Convergence Loss
        sc_loss = torch.norm(y_mag - x_mag, p='fro') / torch.norm(y_mag, p='fro')

        # Log STFT Magnitude Loss
        log_x_mag = torch.log(x_mag + 1e-7)
        log_y_mag = torch.log(y_mag + 1e-7)
        mag_loss = F.l1_loss(log_x_mag, log_y_mag)

        return sc_loss + mag_loss

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: List[int] = [1024, 2048, 512],
        hop_sizes: List[int] = [120, 240, 50],
        win_sizes: List[int] = [600, 1200, 240],
    ):
        super().__init__()
        self.loss_funcs = nn.ModuleList([
            STFTLoss(fft_size, hop_size, win_size)
            for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        losses = [loss_fn(x, y) for loss_fn in self.loss_funcs]
        return sum(losses) / len(losses)
