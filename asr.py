import torchaudio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)
import torch
import re
import sys
import streamlit as st
import sounddevice as sd
from scipy.io import wavfile
from spellchecker import SpellChecker

@st.cache()
def load_model():
    model_name = "facebook/wav2vec2-large-xlsr-53-french"
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    resampler = torchaudio.transforms.Resample(orig_freq=16_000, new_freq=16_000)
    return model, processor, resampler

chars_to_ignore_regex = "[\,\?\.\!\-\;\:\"]"  # noqa: W605
#batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower().replace("’", "'")

def map_to_array(file_list, resampler):
    speech, _ = torchaudio.load(file_list)
    speech = resampler.forward(speech.squeeze(0)).numpy()
    sampling_rate = 16000

    return speech, sampling_rate

def predict(file_list):

    model, processor, resampler = load_model()
    speech, sampling_rate = map_to_array(file_list, resampler)

    features = processor(speech, sampling_rate=sampling_rate, padding=True, return_tensors="pt")
    input_values = features.input_values
    attention_mask = features.attention_mask
    
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    
    pred_ids = torch.argmax(logits, dim=-1)
    predicted = processor.batch_decode(pred_ids)
    
    return predicted

def record_and_predict(sr=16000, channels=1, duration=5, filename='temp.wav'):
    
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=channels).reshape(-1)
    sd.wait()
    wavfile.write("temp.wav", sr, recording)
    return predict("temp.wav")

def spell_check(sentence):

    spell = SpellChecker(language="fr")
    sent = sentence.split()

    new_sent = []
    for word in sent:
        new_sent.append(word)
        misspelled = spell.unknown([word])
        if misspelled != set():
            new_sent.append("(%s)"%spell.correction(word))

    return " ".join(new_sent)