import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

class AudioAmplifier:
    def __init__(self, root):
        self.root = root
        self.root.title("sound amplifier")
        
        # Create GUI elements
        self.select_button = tk.Button(root, text="select WAV file", command=self.select_file)
        self.select_button.pack(pady=10)
        
        tk.Label(root, text="amplification factor:").pack()
        self.amp_factor = tk.Entry(root)
        self.amp_factor.insert(0, "2.0")
        self.amp_factor.pack(pady=5)
        
        self.process_button = tk.Button(root, text="start processing", command=self.process_audio)
        self.process_button.pack(pady=10)
        
        self.status_label = tk.Label(root, text="")
        self.status_label.pack(pady=5)
        
        self.filename = None
        
    def select_file(self):
        self.filename = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav")]
        )
        if self.filename:
            self.status_label.config(text=f"selected file: {self.filename}")
            
    def process_audio(self):
        if not self.filename:
            self.status_label.config(text="please select a file first")
            return
            
        try:
            # Read audio file using soundfile
            audio_data, sample_rate = sf.read(self.filename)
            
            # Convert to float for processing (soundfile already returns float)
            # Get amplification factor
            amp = float(self.amp_factor.get())
            
            # Amplify the signal
            amplified_data = audio_data * amp
            
            # Clip to prevent distortion (-1 to 1 for float)
            amplified_data = np.clip(amplified_data, -1.0, 1.0)
            
            # Plot original and amplified signals
            plt.figure(figsize=(12, 6))
            
            plt.subplot(2, 1, 1)
            plt.plot(audio_data)
            plt.title("original signal")
            plt.ylim(-1, 1)
            plt.xlabel("sample")
            plt.ylabel("amplitude")
            
            plt.subplot(2, 1, 2)
            plt.plot(amplified_data)
            plt.title(f"amplified signal (x{amp})")
            plt.ylim(-1, 1)
            plt.xlabel("sample")
            plt.ylabel("amplitude")
            
            plt.tight_layout()
            plt.show()
            
            # Save amplified audio using soundfile
            output_filename = self.filename.replace(".wav", "_amplified.wav")
            sf.write(output_filename, amplified_data, sample_rate)
            
            self.status_label.config(text=f"processing completed! saved to: {output_filename}")
            
        except Exception as e:
            self.status_label.config(text=f"error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAmplifier(root)
    root.mainloop() 