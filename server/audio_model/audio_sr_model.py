import torch, torch.nn as nn

class AudioSRNet(nn.Module):
    def __init__(self, upsample_factor=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,64,9,padding=4), nn.ReLU(),
            nn.Conv1d(64,128,9,padding=4), nn.ReLU(),
            nn.ConvTranspose1d(128,64,16,stride=upsample_factor,padding=4,output_padding=upsample_factor-1),
            nn.ReLU(),
            nn.Conv1d(64,1,9,padding=4)
        )
    def forward(self,x): return self.net(x)
