import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

# configuração
duracao = 5  # segundos
sample_rate = 16000

print("🎤 Fale algo...")

audio = sd.rec(int(duracao * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()

write("gravacao.wav", sample_rate, audio)

print("Transcrevendo...")

model = WhisperModel("base", device="cpu")

segments, info = model.transcribe("gravacao.wav")

for segment in segments:
    print("Você disse:", segment.text)