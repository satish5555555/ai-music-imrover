# model_service.py
import os
import math
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torchaudio import transforms

# -------------------------
# Utilities
# -------------------------
def load_audio_high_quality(file_path: str, target_sr: int = 48000, max_len: Optional[float] = None, stereo: bool = True):
    waveform, sr = torchaudio.load(file_path)
    # convert to stereo or mono
    if not stereo and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # normalize
    maxv = waveform.abs().max().clamp_min(1e-6)
    waveform = waveform / maxv
    # optionally crop/pad
    if max_len is not None:
        max_samples = int(target_sr * max_len)
        if waveform.shape[1] > max_samples:
            start = 0
            waveform = waveform[:, start:start + max_samples]
        else:
            pad_len = max_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_len))
    return waveform, target_sr

def rms(x: torch.Tensor, eps=1e-9):
    return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)

def match_rms(x: torch.Tensor, ref: torch.Tensor):
    xr = rms(x)
    rr = rms(ref)
    return x * (rr / (xr + 1e-9))

def soft_limiter(x: torch.Tensor, threshold=0.98):
    # soft clipping via tanh to avoid harsh distortion
    x = x / threshold
    return threshold * torch.tanh(x)

# -------------------------
# Model: Multi-scale Residual UNet (1D)
# -------------------------
class ResBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=15, stride=2, padding=7),  # downsample by 2
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            ResBlock1D(out_ch),
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.net = nn.Sequential(
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            ResBlock1D(out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # If length mismatch due to odd sizes, pad or trim
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[:, :, :skip.size(-1)]
        x = x + skip  # residual merge
        return self.net(x)

class MultiScaleUNet(nn.Module):
    """
    Multi-scale 1D U-Net with residual blocks and skip merges.
    Input: (B, C, T) where C typically 2 (stereo) or 1
    Output: same shape as input
    """
    def __init__(self, in_ch=2, base_ch=64, num_scales=5):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=15, padding=7),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True)
        )

        # encoder
        enc_chs = [base_ch * (2**i) for i in range(num_scales)]
        self.downs = nn.ModuleList()
        prev_ch = base_ch
        for ch in enc_chs[1:]:
            self.downs.append(DownBlock(prev_ch, ch))
            prev_ch = ch

        # bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock1D(prev_ch),
            nn.Conv1d(prev_ch, prev_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # decoder (reverse)
        dec_chs = list(reversed(enc_chs[:-1]))
        self.ups = nn.ModuleList()
        for ch in dec_chs:
            self.ups.append(UpBlock(prev_ch, ch))
            prev_ch = ch

        self.out_conv = nn.Sequential(
            nn.Conv1d(base_ch, in_ch, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        # x: (B,C,T)
        x0 = self.in_conv(x)
        skips = [x0]
        cur = x0
        for d in self.downs:
            cur = d(cur)
            skips.append(cur)
        cur = self.bottleneck(cur)
        for u, skip in zip(self.ups, reversed(skips[:-1])):
            cur = u(cur, skip)
        out = self.out_conv(cur)
        return out

# -------------------------
# Multi-resolution STFT loss
# -------------------------
class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss -- spectral convergence + magnitude L1 across multiple FFT sizes.
    """
    def __init__(self, fft_sizes=(1024, 2048, 512), hop_sizes=(120, 240, 50), win_lengths=(600, 1200, 240)):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window_cache = {}

    def _stft(self, x, n_fft, hop_length, win_length, window):
        return torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window.to(x.device), return_complex=True)

    def forward(self, x, y):
        # x,y: (B, C, T)  waveform in [-1,1]
        loss_sc = 0.0
        loss_mag = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            key = (n_fft, win)
            if key not in self.window_cache:
                self.window_cache[key] = torch.hann_window(win)
            win_tensor = self.window_cache[key]

            # compute complex STFT per channel (flatten batch+channel)
            B, C, T = x.shape
            x_ = x.view(B*C, T)
            y_ = y.view(B*C, T)

            X = self._stft(x_, n_fft, hop, win, win_tensor)
            Y = self._stft(y_, n_fft, hop, win, win_tensor)
            # magnitudes
            Xm = torch.abs(X)
            Ym = torch.abs(Y)
            # spectral convergence
            sc = torch.norm(Ym - Xm, p='fro') / (torch.norm(Ym, p='fro') + 1e-9)
            mag_l1 = torch.mean(torch.abs(Ym - Xm))
            loss_sc += sc
            loss_mag += mag_l1
        loss = loss_sc + 0.5 * loss_mag
        return loss

# -------------------------
# Training & Inference
# -------------------------
def train_model(data_dir="/app/server/uploads",
                epochs=30,
                batch_size=2,
                lr=3e-4,
                target_sr=48000,
                segment_seconds=4.0,
                device: Optional[str] = None,
                ckpt_dir: str = None):
    """
    Train model with multi-res STFT loss and waveform L1.
    Uses mixed precision (AMP) if CUDA available.
    """
    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.wav")) + list(data_dir.glob("*.mp3"))
    if not files:
        raise RuntimeError(f"No audio files found in {data_dir}")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir or (Path(__file__).parent / "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = MultiScaleUNet(in_ch=2, base_ch=64, num_scales=5).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stft_loss = MultiResolutionSTFTLoss().to(device)
    l1 = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

    model.train()
    seg_len = int(segment_seconds * target_sr)

    for epoch in range(epochs):
        total_loss = 0.0
        # shuffle files each epoch
        random_order = files.copy()
        import random
        random.shuffle(random_order)
        for idx in range(0, len(random_order), batch_size):
            batch_files = random_order[idx: idx + batch_size]
            wave_batch = []
            for f in batch_files:
                wav, sr = load_audio_high_quality(str(f), target_sr=target_sr, max_len=None, stereo=True)
                # ensure stereo (2,ch)
                if wav.shape[0] == 1:
                    wav = wav.repeat(2, 1)
                # random crop for segment training
                if wav.shape[1] > seg_len:
                    start = random.randint(0, wav.shape[1] - seg_len)
                    wav = wav[:, start:start + seg_len]
                else:
                    wav = F.pad(wav, (0, max(0, seg_len - wav.shape[1])))
                wave_batch.append(wav)
            x = torch.stack(wave_batch, dim=0).to(device)  # (B, C, T)

            # augmentation: small noise + random gain
            noise = 0.0005 * torch.randn_like(x).to(device)
            gains = (torch.randn(x.size(0), 1, 1, device=device) * 0.05 + 1.0)
            noisy = x * gains + noise

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                out = model(noisy)
                # length safety
                if out.size(-1) != x.size(-1):
                    out = F.pad(out, (0, max(0, x.size(-1) - out.size(-1))))
                    out = out[:, :, :x.size(-1)]

                loss_wav = l1(out, x)
                loss_spec = stft_loss(out, x)
                loss = loss_wav * 0.8 + loss_spec * 1.2

            scaler.scale(loss).backward()
            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / math.ceil(len(files) / batch_size)
        print(f"[TRAIN] Epoch {epoch+1}/{epochs}  AvgLoss={avg_loss:.6f}")
        # save periodic checkpoint
        ckpt_path = ckpt_dir / f"model_epoch{epoch+1}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch+1}, ckpt_path)

    # save final
    final_ckpt = ckpt_dir / "trained_model_hifi_multiscale.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"[TRAIN] ✅ Saved final model to {final_ckpt}")

# -------------------------
# Inference — overlap-add, RMS match, soft limiter
# -------------------------
def _overlap_add_infer(model: nn.Module, waveform: torch.Tensor, chunk_size: int, hop: int, device: str):
    """
    waveform: (C, T)
    returns: (C, T)
    """
    model.to(device)
    model.eval()
    C, T = waveform.shape
    pad = (chunk_size - (T % hop)) % hop
    if pad:
        waveform = F.pad(waveform, (0, pad))
        T = waveform.shape[1]

    out = torch.zeros_like(waveform)
    weight = torch.zeros_like(waveform)

    window = torch.hann_window(chunk_size, device=device)

    with torch.no_grad():
        for start in range(0, T - hop + 1, hop):
            seg = waveform[:, start:start + chunk_size].unsqueeze(0).to(device)  # (1,C,L)
            # ensure shape (B,C,T) for model as used in training
            seg = seg.to(device)
            enhanced = model(seg)
            enhanced = enhanced.squeeze(0).cpu()
            # apply window
            win = window.unsqueeze(0)
            out[:, start:start + chunk_size] += enhanced * win
            weight[:, start:start + chunk_size] += win
    weight = weight.clamp_min(1e-9)
    return out / weight

def improve_music_or_audio(input_path: str, output_path: str, mode: str = "auto", target_sr: int = 48000, device: Optional[str] = None):
    """
    Enhance stereo audio track and export high-quality WAV/MP3.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiScaleUNet(in_ch=2, base_ch=64, num_scales=5).to(device)
    ckpt_path = Path(__file__).parent / "checkpoints" / "trained_model_hifi_multiscale.pt"
    if not ckpt_path.exists():
        raise RuntimeError("No trained model found. Please run train_model(...) first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    waveform, sr = load_audio_high_quality(input_path, target_sr=target_sr, stereo=True)
    # ensure shape (C, T)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # inference with overlap-add
    chunk_sec = 3.0
    chunk_size = int(chunk_sec * target_sr)
    hop = chunk_size // 2  # 50% overlap
    enhanced = _overlap_add_infer(model, waveform, chunk_size, hop, device)

    # length match
    enhanced = enhanced[:, :waveform.shape[1]]

    # post-processing: lowpass to remove potential high-band artifacts
    enhanced = torchaudio.functional.lowpass_biquad(enhanced, target_sr, 20000)

    # match RMS of original to preserve perceived loudness
    enhanced = match_rms(enhanced, waveform)

    # soft limiter + small dithering to int16
    enhanced = soft_limiter(enhanced, threshold=0.98)
    enhanced = enhanced / enhanced.abs().max().clamp_min(1e-6)
    int16_audio = (enhanced * 32767.0).short()

    os.makedirs(Path(output_path).parent, exist_ok=True)
    temp_wav = str(Path(output_path).with_suffix(".wav"))
    torchaudio.save(temp_wav, int16_audio, target_sr, format="wav")

    ext = Path(output_path).suffix.lower()
    if ext == ".mp3":
        # Use ffmpeg to create MP3 if requested (must be available)
        os.system(f"ffmpeg -y -i {temp_wav} -codec:a libmp3lame -b:a 320k {output_path} >/dev/null 2>&1")
        os.remove(temp_wav)
    else:
        if not output_path.endswith(".wav"):
            os.rename(temp_wav, output_path)

    print(f"[IMPROVE] ✅ Saved high-fidelity enhanced audio to {output_path}")

# -------------------------
# Quick CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    t = sub.add_parser("train")
    t.add_argument("--data_dir", default="./uploads")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch_size", type=int, default=2)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--target_sr", type=int, default=48000)

    i = sub.add_parser("improve")
    i.add_argument("input_path")
    i.add_argument("output_path")
    i.add_argument("--target_sr", type=int, default=48000)

    args = parser.parse_args()
    if args.cmd == "train":
        train_model(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, target_sr=args.target_sr)
    elif args.cmd == "improve":
        improve_music_or_audio(args.inpu_

