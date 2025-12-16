from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import io

def load_tts():
    model = VitsModel.from_pretrained("facebook/mms-tts-ind")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ind")
    return model, tokenizer

def text_to_speech_bytes(text):
    model, tokenizer = load_tts()
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    audio = output.cpu().numpy().squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()
    wavfile.write(buffer, model.config.sampling_rate, audio)
    buffer.seek(0)

    return buffer.read()

def text_to_speech(text, path='test.wav'):
    model, tokenizer = load_tts()
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    audio = output.detach().cpu().numpy().squeeze(0)
    audio = np.clip(audio, -1.0, 1.0)

    audio = (audio * 32767).astype(np.int16)
    wavfile.write(
        path,
        rate=model.config.sampling_rate,
        data=audio
    )
