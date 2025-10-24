import pretty_midi
def beam_autocorrect(input_midi, out_midi, cfg, beam_width=4):
    # simple quantize & smoothing
    pm = pretty_midi.PrettyMIDI(input_midi)
    time_step = cfg.get("data",{}).get("time_step", 0.125)
    for instr in pm.instruments:
        for n in instr.notes:
            n.start = round(n.start/time_step)*time_step
            n.end = round(n.end/time_step)*time_step
    pm.write(out_midi)
