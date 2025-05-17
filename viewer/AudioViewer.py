import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

class WaveViewer:
    """
    Class to load audio files and visualize them in time and frequency domains.
    """
    def __init__(self, audio_path: str, channel_mode: str = 'stereo'):
        audio_signal, fs = sf.read(audio_path)
        
        # Check if the audio is mono or stereo
        if len(audio_signal.shape) == 1:  # Mono
            self.d = audio_signal
            self.x = audio_signal if channel_mode == 'mono' else None
        else:  # Stereo
            self.d = audio_signal[:, 0]
            self.x = audio_signal[:, 1] if channel_mode == 'stereo' else audio_signal[:, 0]
            
        self.fs_d = fs
        self.fs_x = fs
        
        # clip original audio
        start_clip = 0
        self.d = self.d[start_clip:]
        if self.x is not None:
            self.x = self.x[start_clip:]
            self.x = self.x * 1.0

        self.N = len(self.d)

        # check sampling rate (strictly same is desirable)
        if self.fs_d != self.fs_x:
            print("[Warning] Primary and Reference sampling frequencies are different.")

        # 信号の統計情報を計算
        self.d_stats = {
            'max': np.max(np.abs(self.d)),
            'rms': np.sqrt(np.mean(self.d**2)),
            'peak_to_peak': np.max(self.d) - np.min(self.d),
            'db_rms': 20 * np.log10(np.sqrt(np.mean(self.d**2))),  # RMSベースのdB
            'db_peak': 20 * np.log10(np.max(np.abs(self.d)))       # ピークベースのdB
        }
        
        if self.x is not None:
            self.x_stats = {
                'max': np.max(np.abs(self.x)),
                'rms': np.sqrt(np.mean(self.x**2)),
                'peak_to_peak': np.max(self.x) - np.min(self.x),
                'db_rms': 20 * np.log10(np.sqrt(np.mean(self.x**2))),  # RMSベースのdB
                'db_peak': 20 * np.log10(np.max(np.abs(self.x)))       # ピークベースのdB
            }

    def plot_signals_time_domain(self, start_sample=0, num_samples=None, ax=None):
        """
        Visualize input and output signals in time domain
        """
        if num_samples is None:
            end_sample = self.N
        else:
            end_sample = min(start_sample + num_samples, self.N)
            
        time_axis = np.arange(start_sample, end_sample) / self.fs_d
        
        if ax is None:
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
        elif isinstance(ax, np.ndarray):  # ax is a numpy array
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)
        
        ax.plot(time_axis, self.d[start_sample:end_sample], label='main signal', alpha=0.8)
        if self.x is not None:
            ax.plot(time_axis, self.x[start_sample:end_sample], label='sub signal', alpha=0.8)
        ax.set_title('main signal and output signal')
        ax.set_xlabel('time [sec]')
        ax.set_ylabel('amplitude')
        ax.legend()
        ax.grid(True)
        
        # show statistics as text
        stats_text = (f'Main signal stats:\n'
                     f'Peak dB: {self.d_stats["db_peak"]:.1f} dBFS\n'
                     f'RMS dB: {self.d_stats["db_rms"]:.1f} dBFS')
        
        if self.x is not None:
            stats_text += (f'\n\nSub signal stats:\n'
                         f'Peak dB: {self.x_stats["db_peak"]:.1f} dBFS\n'
                         f'RMS dB: {self.x_stats["db_rms"]:.1f} dBFS')
        
        ax.text(1.02, 0.5, stats_text,
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='center')
        
        return ax

    def plot_signals_frequency_domain(self, nfft=2048):
        """
        Visualize input and output signals in frequency domain
        
        Parameters
        ----------
        nfft : int
            FFTのサイズ
        """
        from scipy import signal
        
        # calculate frequency domain
        f, Pxx_d = signal.welch(self.d, self.fs_d, nperseg=nfft)
        _, Pxx_x = signal.welch(self.x, self.fs_x, nperseg=nfft)

        # convert to dB scale
        Pxx_d_db = 10 * np.log10(Pxx_d)
        Pxx_x_db = 10 * np.log10(Pxx_x)
        
        plt.figure(figsize=(14, 10))
        
        # plot main signal spectrum
        plt.subplot(3, 1, 1)
        plt.semilogx(f, Pxx_d_db)
        plt.title('Main signal spectrum')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power [dB/Hz]')
        plt.grid(True)
        plt.xlim([20, self.fs_d/2])
        
        # plot reference signal spectrum
        plt.subplot(3, 1, 2)
        plt.semilogx(f, Pxx_x_db)
        plt.title('Reference signal spectrum')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power [dB/Hz]')
        plt.grid(True)
        plt.xlim([20, self.fs_d/2])
         
        plt.tight_layout()
        plt.show()

    def error_plot(self):
        # align the length of dimensions and then subtract
        d_ = self.d
        x_ = self.x
        if len(self.d) > len(self.x):
            d_ = self.d[:len(self.x)]
        else:
            x_ = self.x[:len(self.d)]
        error = d_ - x_

        plt.figure(figsize=(10, 6))
        plt.plot(error)
        plt.title('error signal')
        plt.xlabel('time [sec]')
        plt.ylabel('amplitude')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def select_and_plot_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Create channel mode selection dialog
    channel_dialog = tk.Toplevel(root)
    channel_dialog.title("Select channel mode")
    channel_dialog.geometry("300x150")

    channel_mode = tk.StringVar(value="stereo")
    
    ttk.Radiobutton(channel_dialog, text="Stereo", variable=channel_mode, value="stereo").pack(pady=10)
    ttk.Radiobutton(channel_dialog, text="Mono", variable=channel_mode, value="mono").pack(pady=10)

    def proceed():
        selected_mode = channel_mode.get()
        channel_dialog.destroy()
        
        files = filedialog.askopenfilenames(
            title="Select WAV files",
            filetypes=[("WAV files", "*.wav")]
        )
        
        if files:
            # Create subplot grid
            n_files = len(files)
            n_rows = (n_files + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(20, 5*n_rows))
            
            # axesを1次元配列に変換
            axes = axes.flatten()
            
            # first read all data and then find the maximum and minimum values of the y-axis
            y_min = float('inf')
            y_max = float('-inf')
            viewers = []
            
            for file in files:
                viewer = WaveViewer(file, channel_mode=selected_mode)
                viewers.append(viewer)
                y_min = min(y_min, np.min(viewer.d))
                y_max = max(y_max, np.max(viewer.d))
                if viewer.x is not None:
                    y_min = min(y_min, np.min(viewer.x))
                    y_max = max(y_max, np.max(viewer.x))
            
            # plot each file
            for i, (file, viewer) in enumerate(zip(files, viewers)):
                viewer.plot_signals_time_domain(ax=axes[i])
                axes[i].set_title(f'File: {file.split("/")[-1]}')
                axes[i].set_ylim([y_min, y_max])
            
            # Hide empty subplots
            for i in range(n_files, n_rows * 2):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        
        root.destroy()

    ttk.Button(channel_dialog, text="Continue", command=proceed).pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    select_and_plot_files()
    
