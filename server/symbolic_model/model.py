import torch, torch.nn as nn, math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))
        pe[:,0::2], pe[:,1::2] = torch.sin(pos*div), torch.cos(pos*div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])

class MusicTransformer(nn.Module):
    def __init__(self, vocab, cfg):
        super().__init__()
        d = cfg["model"]
        self.embed = nn.Embedding(vocab, d["d_model"])
        self.pos = PositionalEncoding(d["d_model"], d["dropout"])
        layer = nn.TransformerEncoderLayer(d["d_model"], d["nhead"], d["dim_feedforward"], d["dropout"], batch_first=False)
        self.encoder = nn.TransformerEncoder(layer, d["num_layers"])
        self.dec = nn.Linear(d["d_model"], vocab)

    def _mask(self, n, dev):
        return torch.triu(torch.full((n,n), float('-inf')),1).to(dev)

    def forward(self, x):
        dev = x.device
        m = self._mask(x.size(1), dev)
        x = self.embed(x).permute(1,0,2)
        x = self.pos(x)
        out = self.encoder(x, mask=m)
        return self.dec(out.permute(1,0,2))
