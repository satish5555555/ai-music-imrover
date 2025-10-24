import os, mimetypes
from symbolic_model.generate import improve_midi_sync
from audio_model.enhance_audio import improve_audio_sync

def improve_music_or_audio(input_path, output_path, mode="auto"):
    mime = mimetypes.guess_type(input_path)[0] or ""
    if mode == "music" or input_path.lower().endswith((".mid", ".midi")):
        improve_midi_sync(input_path, output_path)
    elif mode == "audio" or "audio" in mime or input_path.lower().endswith((".wav", ".mp3")):
        improve_audio_sync(input_path, output_path)
    else:
        raise Exception("Unsupported file format")
