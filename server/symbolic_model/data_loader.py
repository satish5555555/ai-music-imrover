import os
import numpy as np
import pretty_midi
from tqdm import tqdm

def midi_to_frames(path, time_step=0.125, min_pitch=21, max_pitch=108):
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception:
        return []
    max_time = pm.get_end_time()
    if max_time <= 0:
        return []
    times = np.arange(0, max_time + time_step, time_step)
    pr = pm.get_piano_roll(fs=int(1/time_step)) if max_time>0 else np.zeros((128,1))
    frames = []
    for t in times:
        idx = min(int(round(t * pr.shape[1] / max(1, max_time + 1e-8))), pr.shape[1]-1)
        col = pr[:, idx]
        col = col[min_pitch:max_pitch+1]
        col = (col > 0).astype(np.uint8)
        token = int.from_bytes(col.tobytes(), byteorder='little')
        frames.append(token)
    return frames
