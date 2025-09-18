import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
import sounddevice as sd

fs = 44100   
seconds = 2  
filename = "myvoice.wav"

print("Recording for 2 seconds...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
write(filename, fs, recording)  
print(f"Recording finished and saved as {filename}")

fs, data = read(filename)
print("Sample rate:", fs, "Hz")
print("Number of samples:", len(data))

if data.ndim > 1:
    data = data[:, 0]

time_axis = np.arange(len(data)) / fs
plt.figure(figsize=(10, 4))
plt.plot(time_axis, data)
plt.title("Audio Waveform")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

N = len(data)
fft_data = np.fft.fft(data)
fft_freq = np.fft.fftfreq(N, 1/fs)

plt.figure(figsize=(10, 4))
plt.plot(fft_freq[:N//2], np.abs(fft_data[:N//2]))
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
