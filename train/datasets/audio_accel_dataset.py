import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.audio_dataset import AudioDataset
from datasets.accel_dataset_mel import AccelDatasetMel
from datasets.dataset_scale import DatasetScale

class AudioAccelDataset(Dataset):
    def __init__(self, root_dir, transform=None, duration=4, sr=22050, sampling_interval=0.001, 
                 scale=DatasetScale.FULL, output_direction=False, output_velocity=False):  # 出力フラグを追加
        
        self.output_direction = output_direction
        self.output_velocity = output_velocity
        
        self.audio_dataset = AudioDataset(
            root_dir, 
            transform=None, 
            duration=duration, 
            sr=sr, 
            scale=scale, 
            output_direction=output_direction,  # pass flag
            output_velocity=output_velocity     # pass flag
        )
        
        self.accel_dataset = AccelDatasetMel(
            root_dir, 
            transform=None, 
            sampling_interval=sampling_interval, 
            duration=duration, 
            scale=scale
        )

        # Ensure both datasets are of same length and have matching labels
        assert len(self.audio_dataset) == len(self.accel_dataset), "Audio and Accel datasets should be of the same size"
        assert all(a == b for a, b in zip(self.audio_dataset.labels_list, self.accel_dataset.labels_list)), "Mismatch in labels between Audio and Accel datasets"

        self.transform = transform
        
        self.label_to_int = self.audio_dataset.label_to_int
        self.classes = self.audio_dataset.classes

    def __len__(self):
        return len(self.audio_dataset)
    
    def __getitem__(self, idx):
        # get data from sound_dataset
        audio_outputs = self.audio_dataset[idx]
        accel_data, accel_label = self.accel_dataset[idx]
        
        # unpack sound_outputs based on its length
        if len(audio_outputs) == 4:  # (data, label, direction, velocity)
            audio_data, audio_label, direction, velocity = audio_outputs
        else:  # (data, label)
            audio_data, audio_label = audio_outputs
            direction = None
            velocity = None
        
        # Assert that labels from both datasets match for the given index
        assert audio_label == accel_label, f"Labels mismatch at index {idx}"
        
        label = audio_label

        # Apply transform
        if self.transform:
            audio_data = self.transform(audio_data)
            accel_data = self.transform(accel_data)

        # build outputs based on flags
        outputs = [audio_data, accel_data, label]
        if self.output_direction:
            outputs.append(direction)
        if self.output_velocity:
            outputs.append(velocity)
        
        return tuple(outputs)

# display log-mel spectrogram as RGB image
def display_rgb_spectrogram(log_mel_data):
    # normalize to 0-1 range
    log_mel_data_normalized = (log_mel_data - log_mel_data.min()) / (log_mel_data.max() - log_mel_data.min())

    # map 3 channels to RGB
    rgb_image = np.stack([log_mel_data_normalized[0], log_mel_data_normalized[1], log_mel_data_normalized[2]], axis=-1)

    plt.figure(figsize=(10, 4))
    plt.imshow(rgb_image, aspect='auto', interpolation='none')
    plt.title('RGB Log-Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# test
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))
    ])

    dataset = AudioAccelDataset(root_dir="/workspace/texture_dataset", transform=transform, duration=2, sr=22050, sampling_interval=0.001, scale=DatasetScale.LITE)
    
    num_samples = len(dataset)
    print("Number of samples in dataset:", num_samples)

    # get random sample
    import random
    idx = random.randint(0, num_samples-1)

    # get random sample
    audio_data, accel_data, label = dataset[idx]
    print("Audio data shape:", audio_data.shape)
    print("Accel data shape:", accel_data.shape)
    print("Label:", label)


    # plot data
    plt.figure(figsize=(15,5))
    plt.title("mel spectrogram")
    # squeeze
    audio_data = audio_data.squeeze(0)
    plt.imshow(audio_data, aspect='auto', origin='lower')
    plt.colorbar()
    plt.show()

    # squeeze all dimensions
    accel_data = accel_data.squeeze()
    display_rgb_spectrogram(accel_data)
