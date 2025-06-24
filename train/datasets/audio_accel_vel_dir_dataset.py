import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.audio_vel_dir_dataset import AudioVelDirDataset
from datasets.accel_vel_dir_dataset import AccelVelDirDataset


class AudioAccelVelDirDataset(Dataset):
    def __init__(self, index, root_dir, transform=None, label="vel", duration=4, sr=22050, sampling_interval=0.001):
        self.audio_dataset = AudioVelDirDataset(index, root_dir, label, transform=None, duration=duration, sr=sr)
        self.accel_dataset = AccelVelDirDataset(index, root_dir, label, transform=None, sampling_interval=sampling_interval, duration=duration)

        # Ensure both datasets are of same length and have matching labels
        assert len(self.audio_dataset) == len(self.accel_dataset), "Audio and Accel datasets should be of the same size"
        assert all(a == b for a, b in zip(self.audio_dataset.labels_list, self.accel_dataset.labels_list)), "Mismatch in labels between Audio and Accel datasets"

        self.transform = transform
        
        self.label_to_int = self.audio_dataset.label_to_int
        self.classes = self.audio_dataset.classes

        self.data_list = []

    def __len__(self):
        return len(self.audio_dataset)
    
    def __getitem__(self, idx):
        audio_data, audio_label = self.audio_dataset[idx]
        accel_data, accel_label = self.accel_dataset[idx]
        
        # Assert that labels from both datasets match for the given index
        assert audio_label == accel_label, f"Labels mismatch at index {idx}"
        
        if self.transform:
            audio_data = self.transform(audio_data)
            accel_data = self.transform(accel_data)

        return audio_data, accel_data, audio_label
    

# test
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 1024))
    ])

    dataset = AudioAccelVelDirDataset(index=0, root_dir="/workspace/texture_dataset", transform=transform, label="vel", duration=2, sr=22050, sampling_interval=0.001)
    
    num_samples = len(dataset)
    print("Number of samples in dataset:", num_samples)

    # get random sample
    idx = np.random.randint(num_samples)
    sample, label = dataset[idx]
    print("Sample shape:", sample.shape)
    print("Label:", label)
    print("Label name:", dataset.audio_dataset.classes[label])

    audio_data = sample[0]
    accel_data = sample[1]

    # plot data
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,5))
    plt.title("Audio and Accel data")
    plt.subplot(2, 1, 1)
    plt.title("Audio data")
    plt.imshow(audio_data.squeeze(0), cmap='hot', aspect='auto')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.title("Accel data")
    plt.imshow(accel_data.squeeze(0), cmap='hot', aspect='auto')
    plt.colorbar()
    plt.show()
