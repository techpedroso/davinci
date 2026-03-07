import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import ollama

from faster_whisper import WhisperModel
from piper import PiperVoice

# -------------------------
# carregar modelos
# -------------------------

print("Carregando modelos...")

whisper = WhisperModel("base", device="cpu", compute_type="int8_float32")
voice = PiperVoice.load("voice/model.onnx")

print("Pronto.")

# -------------------------
# função falar
# -------------------------

def speak(text):

    print("Jarvis:", text)

    chunks = []

    for audio in voice.synthesize(text):
        chunks.append(audio.audio_float_array)
        sample_rate = audio.sample_rate

    audio_all = np.concatenate(chunks)

    sd.play(audio_all, sample_rate)
    sd.wait()


# -------------------------
# função ouvir
# -------------------------

def listen():

    print("Ouvindo...")

    duration = 5
    samplerate = 16000

    recording = sd.rec(int(duration * samplerate),
                       samplerate=samplerate,
                       channels=1)

    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, recording, samplerate)

        segments, _ = whisper.transcribe(f.name)

        text = " ".join([seg.text for seg in segments])

    return text.strip()


# -------------------------
# loop principal
# -------------------------

print("Jarvis iniciado.")

while True:

    text = listen()

    if not text:
        continue

    print("Você:", text)

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": text}]
    )

    reply = response["message"]["content"]

    speak(reply)