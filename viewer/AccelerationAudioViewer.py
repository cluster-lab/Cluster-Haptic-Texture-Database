import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def open_files():
    # show dialog to select CSV file
    csv_path = filedialog.askopenfilename(
        title="Select CSV(acceleration) file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    
    # show dialog to select WAV file
    wav_path = filedialog.askopenfilename(
        title="Select WAV(audio) file",
        filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
    )
    
    if csv_path and wav_path:
        # read CSV file
        data = pd.read_csv(csv_path)
        
        # read WAV file
        y, sr = librosa.load(wav_path)
        time_audio = np.linspace(0, len(y)/sr, len(y))
        
        # sum of acceleration data (3 axes)
        acc_total = np.sqrt(data['X']**2 + data['Y']**2 + data['Z']**2)
        
        # create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # plot acceleration data
        ax1.plot(data['time'], acc_total, label="Total Acceleration")
        ax1.plot(data['time'], data['X'], label="X-axis Acceleration")
        ax1.plot(data['time'], data['Y'], label="Y-axis Acceleration")
        ax1.plot(data['time'], data['Z'], label="Z-axis Acceleration")
        ax1.set_title("3D Acceleration Data")
        ax1.set_ylabel("Acceleration")
        ax1.legend()
        ax1.grid(True)
        
        # plot audio data
        ax2.plot(time_audio, y)
        ax2.set_title("Audio Signal")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

# create GUI
root = tk.Tk()
root.title("Acceleration and Audio Plotter")

# select file button
button = tk.Button(root, text="Load CSV and WAV", command=open_files)
button.pack(padx=20, pady=20)

root.mainloop() 