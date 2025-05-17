import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tkinter as tk
from tkinter import filedialog

def plot_audio_analysis(wav_file):
    # Load the audio file
    y, sr = librosa.load(wav_file, sr=None, mono=False)
    print(f"Sample Rate: {sr}")

    # If stereo, process each channel separately
    if y.ndim == 2:
        print("Stereo")
        for channel in range(y.shape[0]):
            plot_channel_analysis(y[channel], sr, channel)
    else:
        print("Monoral")
        plot_channel_analysis(y, sr)

def plot_channel_analysis(y, sr, channel=None):
    # Perform FFT
    fft_result = np.fft.fft(y)
    freq = np.fft.fftfreq(len(y), 1 / sr)

    # Only take the positive half of the frequencies and magnitudes
    positive_freqs = freq[:len(freq)//2]
    positive_magnitudes = np.abs(fft_result)[:len(fft_result)//2]

    # Create a figure with subplots
    fig, ax = plt.subplots(3, 1, figsize=(12, 18))

    # Plot the time-domain signal
    librosa.display.waveshow(y[:10000], sr=sr, ax=ax[0])
    title = 'Time-Domain Signal'
    if channel is not None:
        title += f' (Channel {channel})'
    ax[0].set_title(title)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude')

    # Plot the FFT result
    ax[1].plot(positive_freqs, positive_magnitudes)
    title = 'FFT of Audio Signal'
    if channel is not None:
        title += f' (Channel {channel})'
    ax[1].set_title(title)
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Magnitude')
    ax[1].set_xlim([0, 2000])  # Limit the frequency range for better visualization
    ax[1].grid()

    # Plot the spectrogram
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax[2])
    fig.colorbar(img, ax=ax[2], format='%+2.0f dB')
    title = 'Spectrogram'
    if channel is not None:
        title += f' (Channel {channel})'
    ax[2].set_title(title)
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()

def select_file_and_plot_analysis():
    # Open file dialog to select a file
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

    # Plot the analysis if a file was selected
    if file_path:
        plot_audio_analysis(file_path)

# Run the file selection and plotting function
select_file_and_plot_analysis()