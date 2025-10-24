# Placeholder training script for symbolic model
import argparse, yaml, torch, time, os
from data_loader import midi_to_frames
from model import MusicTransformer
from torch.utils.data import DataLoader, Dataset
import numpy as np

class SimpleSeqDataset(Dataset):
    def __init__(self, sequences, seq_len):
        self.seq = sequences
        self.seq_len = seq_len
    def __len__(self): return len(self.seq)
    def __getitem__(self, idx):
        s = self.seq[idx]
        x = np.array(s[:-1], dtype=np.int64)
        y = np.array(s[1:], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

def load_data(midi_dir, seq_len):
    seqs = []
    for root,_,files in os.walk(midi_dir):
        for f in files:
            if f.lower().endswith((".mid",".midi")):
                frames = midi_to_frames(os.path.join(root,f))
                if len(frames) > seq_len:
                    for i in range(0,len(frames)-seq_len, seq_len):
                        seqs.append(frames[i:i+seq_len+1])
    return seqs

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    seq_len = cfg["training"]["seq_len"]
    data = load_data(cfg["data"]["midi_dir"], seq_len)
    if len(data)==0:
        print("No MIDI data found. Place MIDI files in server/training_data/midi/")
        return
    vocab = 4096  # placeholder; for real use build vocab map
    model = MusicTransformer(vocab, cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    for epoch in range(cfg["training"]["epochs"]):
        t0 = time.time()
        for x,y in DataLoader(SimpleSeqDataset(data, seq_len), batch_size=cfg["training"]["batch_size"], shuffle=True):
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optim.zero_grad(); loss.backward(); optim.step()
        print(f"Epoch {epoch+1} done")
    torch.save({"model_state": model.state_dict()}, "../checkpoints/symbolic_best.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); args=p.parse_args()
    cfg = yaml.safe_load(open(args.config)); train(cfg)
