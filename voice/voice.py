from piper import PiperVoice
import sounddevice as sd

voice = PiperVoice.load("model.onnx")

for audio in voice.synthesize("Olá. Eu sou o DaVinci. É um prazer conversar com você!"):
    sd.play(audio.audio_float_array, audio.sample_rate)
    sd.wait()