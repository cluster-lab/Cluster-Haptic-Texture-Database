import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import noisereduce as nr

class NoiseReduceFilter:
    def __init__(self, mode='stationary', n_fft=2048, hop_length=512, win_length=None,
                 n_std_thresh_stationary=1.5, prop_decrease=1.0,
                 time_constant_s=2.0, freq_mask_smooth_hz=500, time_mask_smooth_ms=50):
        self.mode = mode
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length else n_fft
        self.n_std_thresh_stationary = n_std_thresh_stationary
        self.prop_decrease = prop_decrease
        self.time_constant_s = time_constant_s
        self.freq_mask_smooth_hz = freq_mask_smooth_hz
        self.time_mask_smooth_ms = time_mask_smooth_ms

    def apply(self, data, sr, noise_data=None):
        return nr.reduce_noise(y=data, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                               n_std_thresh_stationary=self.n_std_thresh_stationary, prop_decrease=self.prop_decrease,
                               time_constant_s=self.time_constant_s, freq_mask_smooth_hz=self.freq_mask_smooth_hz,
                               time_mask_smooth_ms=self.time_mask_smooth_ms, stationary=self.mode == 'stationary', y_noise=noise_data)

    def plot_spectrogram(self, data, sr, title="log-mel spectgram", save_path=None):
        plt.figure(figsize=(10, 4))
        S = librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        librosa.display.specshow(S_db, sr=sr, hop_length=self.hop_length, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        if save_path:
            plt.savefig(save_path)
        plt.close()

# test
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog

    # Open file dialog to select an audio file
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

    # Check if a file was selected
    if not file_path:
        print("No file selected")
        exit()

    # Load audio file
    data, sr = librosa.load(file_path, sr=None)

    # Open file dialog to select a noise profile file
    noise_file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")], title="Select Noise Profile File")

    # Load noise profile if selected
    noise_data = None
    if noise_file_path:
        noise_data, noise_sr = librosa.load(noise_file_path, sr=sr)

    # Apply noise reduction
    noise_reduce_filter = NoiseReduceFilter(mode='stationary', n_fft=2048, hop_length=512, win_length=None, n_std_thresh_stationary=2.0, prop_decrease=1.0, time_constant_s=2.0, freq_mask_smooth_hz=500, time_mask_smooth_ms=50)
    reduced_data = noise_reduce_filter.apply(data, sr, noise_data=noise_data)

    # Plot spectrogram
    noise_reduce_filter.plot_spectrogram(data, sr, title="Original Spectrogram")
    noise_reduce_filter.plot_spectrogram(reduced_data, sr, title="Reduced Spectrogram")

    # Save reduced audio to wav file
    output_file_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
    if output_file_path:
        sf.write(output_file_path, reduced_data, sr)
        print(f"File saved to {output_file_path}")
    else:
        print("No output file selected")