import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt

def open_file():
    # show dialog to select CSV(acceleration) file
    file_path = filedialog.askopenfilename(
        title="Select CSV(acceleration) file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    # if file is selected, read and plot
    if file_path:
        # read CSV file (with header)
        data = pd.read_csv(file_path)
        
        # plot with Matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(data['time'], data['X'], label="X")
        plt.plot(data['time'], data['Y'], label="Y")
        plt.plot(data['time'], data['Z'], label="Z")
        plt.title("3D Acceleration Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration")
        plt.legend()
        plt.grid(True)
        plt.show()

# create GUI
root = tk.Tk()
root.title("3D Acceleration Plotter")

# select file button
button = tk.Button(root, text="Load CSV(acceleration)", command=open_file)
button.pack(padx=20, pady=20)

root.mainloop()