import os
from torch.utils.data import Dataset
from torchvision import transforms
import librosa
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.dataset_scale import DatasetScale


class AudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, duration=4, sr=22050, scale=DatasetScale.FULL, output_direction=False, output_velocity=False): 
        self.audio_dir = os.path.join(root_dir, "sensor_data/audio")
        self.output_direction = output_direction
        self.output_velocity = output_velocity

        self.transform = transform

        self.duration = duration  # seconds
        self.sr = sr
        self.total_samples = sr * duration

        folder_list = [os.path.join(self.audio_dir, f) for f in os.listdir(self.audio_dir) if os.path.isdir(os.path.join(self.audio_dir, f))]
        folder_list = sorted(folder_list, key=lambda x: int(os.path.basename(x)))  # sort by texture id

        # scale
        if scale == DatasetScale.LITE:
            folder_list = folder_list[:DatasetScale.LITE.value]
        elif scale == DatasetScale.MEDIAN:
            folder_list = folder_list[:DatasetScale.MEDIAN.value]

        file_list = []
        for folder in folder_list:
            file_list.extend([f for f in os.listdir(folder) if f.endswith('.wav')])

        self.file_list = file_list  # ファイルリストを保存

        # get all texture ids
        all_texture_ids = [self.parse_filename(f)[0] for f in file_list]
        self.label_to_int = self._convert_labels_to_int(all_texture_ids)
        self.classes = list(self.label_to_int.keys())
        # print(self.classes)
        
        # list of data and labels
        self.data_list = []
        self.labels_list = []
        
        print("Loading AudioDataset...")
        total_files = len(file_list)

        # initialize min and max values
        self.min_direction = float('inf')
        self.max_direction = float('-inf')
        self.min_velocity = float('inf')
        self.max_velocity = float('-inf')

        # calculate min and max values for direction and velocity
        if output_direction or output_velocity:
            for file_name in file_list:
                _, direction, velocity, _, _ = self.parse_filename(file_name)
                direction = int(direction)
                velocity = int(velocity)
                
                if output_direction:
                    self.min_direction = min(self.min_direction, direction)
                    self.max_direction = max(self.max_direction, direction)
                
                if output_velocity:
                    self.min_velocity = min(self.min_velocity, velocity)
                    self.max_velocity = max(self.max_velocity, velocity)

        for idx, file_name in enumerate(file_list):
            texture_id, _, _, _, _ = self.parse_filename(file_name)

            audio_path = os.path.join(self.audio_dir, texture_id, file_name)
            audio_data = self.load_wav(audio_path)
            
            self.data_list.append(audio_data)
            self.labels_list.append(self.label_to_int[texture_id])

            # print progress
            progress = (idx + 1) / total_files * 100
            print(f"Progress: {progress:.2f}% ({idx + 1}/{total_files})", end='\r')
        print("")

    def parse_filename(self, filename):
        parts = filename.split('_')
        texture_id, line_degree, velocity, target_force, test_count = parts[0], parts[1], parts[2], parts[3], parts[4].split('.')[0]
        return texture_id, line_degree, velocity, target_force, test_count
    
    def load_wav(self, path):
        waveform, sr = librosa.load(path, sr=self.sr)

        if len(waveform) < self.total_samples:
            # pad waveform with zeros
            waveform = np.pad(waveform, (0, self.total_samples - len(waveform)))

        # log-mel spectrogram
        feature_melspec = librosa.feature.melspectrogram(y=waveform[:self.total_samples], sr=sr, n_mels=128)
        feature_melspec_db = librosa.power_to_db(feature_melspec, ref=np.max)

        return feature_melspec_db

    def _convert_labels_to_int(self, labels_list):
        unique_labels = sorted(set(labels_list))
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        return label_to_int

    def normalize_value(self, value, min_val, max_val):
        """normalize value to 0-1 range"""
        if min_val == max_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        # apply transform
        if self.transform:
            data = self.transform(data)

        label = self.labels_list[idx]

        # return data and label if no additional outputs are needed
        if not self.output_direction and not self.output_velocity:
            return data, label

        # process if direction and velocity are needed
        filename = os.path.basename(self.file_list[idx])
        _, direction, velocity, _, _ = self.parse_filename(filename)
        
        outputs = [data, label]
        if self.output_direction:
            normalized_direction = self.normalize_value(
                int(direction), 
                self.min_direction, 
                self.max_direction
            )
            outputs.append(normalized_direction)
        if self.output_velocity:
            normalized_velocity = self.normalize_value(
                int(velocity), 
                self.min_velocity, 
                self.max_velocity
            )
            outputs.append(normalized_velocity)
        
        return tuple(outputs)
    

# test
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 64)),
        # transforms.Resize((224, 224))
    ])

    dataset = AudioDataset("/workspace/texture_dataset", transform=transform, duration=1, scale=DatasetScale.LITE, output_direction=True, output_velocity=True)

    num_samples = len(dataset)
    print("Number of samples in dataset:", num_samples)
    print("Classes:", dataset.classes)

    # get random sample
    import random
    idx = random.randint(0, num_samples-1)

    # get random sample
    data, label, direction, velocity = dataset[idx]
    print("Data shape:", data.shape)
    print("file name:", dataset.file_list[idx])
    print("Label:", label)
    print("Direction:", direction)
    print("Velocity:", velocity)

    # plot data
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,5))
    plt.title("mel spectrogram")
    # squeeze
    data = data.squeeze(0)
    # rgb to gray
    # data = data.mean(axis=0)
    plt.imshow(data, aspect='auto', origin='lower')
    #plt.imshow(data, aspect='auto', origin='lower')
    plt.colorbar()
    plt.show()
