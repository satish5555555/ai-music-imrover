import torch, torchaudio, os
from .audio_sr_model import AudioSRNet

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

def improve_audio_sync(input_audio, output_audio):
    device = get_device()
    model = AudioSRNet().to(device)
    # (Optionally) load pretrained weights if available at ../checkpoints/audio_sr_best.pt
    # if os.path.exists("../checkpoints/audio_sr_best.pt"):
    #     ck = torch.load("../checkpoints/audio_sr_best.pt", map_location=device); model.load_state_dict(ck["model_state"])
    wav, sr = torchaudio.load(input_audio)
    # convert to mono
    if wav.size(0) > 1: wav = wav.mean(dim=0, keepdim=True)
    wav = wav.to(device)
    with torch.no_grad():
        # reshape: (batch, channel, time)
        y = model(wav.unsqueeze(0)).squeeze(0).cpu()
    # save with upsampled sample rate (sr * upsample factor)
    torchaudio.save(output_audio, y, sr * 4)
