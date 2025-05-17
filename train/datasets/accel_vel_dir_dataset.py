import os
import pandas as pd
import numpy as np
import librosa
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                
from datasets.dataset_scale import DatasetScale


class AccelVelDirDataset(Dataset):
    def __init__(self, index, root_dir, label="vel", transform=None, sampling_interval=0.001, duration=4):
        self.accel_dir = os.path.join(root_dir, "sensor_data/accel")

        self.transform = transform

        self.sampling_interval = sampling_interval
        self.duration = duration  # seconds
        
        folder_list = [os.path.join(self.accel_dir, f) for f in os.listdir(self.accel_dir) if os.path.isdir(os.path.join(self.accel_dir, f))]
        folder_list = sorted(folder_list, key=lambda x: int(os.path.basename(x)))  # sort by texture id
        file_list = []
        for folder in folder_list:
            file_list.extend([f for f in os.listdir(folder) if f.endswith('.csv')])

        # get all texture_ids
        all_texture_ids = sorted([int(self.parse_filename(f)[0]) for f in file_list])
        # create a list of unique texture_ids
        unique_texture_ids = [str(id) for id in sorted(set(all_texture_ids))]
        # get target texture_id based on the index
        target_texture_id = unique_texture_ids[index]
        # get data for the target texture_id
        self.file_list = [f for f in file_list if self.parse_filename(f)[0] == target_texture_id]
        self.file_list = sorted(self.file_list)


        # get all feedrates
        if label == "vel":
            all_label = [self.parse_filename(f)[2] for f in self.file_list]
        elif label == "dir":
            all_label = [self.parse_filename(f)[1] for f in self.file_list]
        self.label_to_int = self._convert_labels_to_int(all_label)
        self.classes = list(self.label_to_int.keys())

        # save data and labels
        self.data_list = []
        self.labels_list = []
        for file_name in self.file_list:
            texture_id, line_degree, velocity, _, _ = self.parse_filename(file_name)

            accel_path = os.path.join(self.accel_dir, texture_id, file_name)
            accel_data = self.load_csv(accel_path).astype(np.float32)  # データ型をfloat32に変換

            if label == "vel":
                mapped_id = self.label_to_int[velocity]
            elif label == "dir":
                mapped_id = self.label_to_int[line_degree]
            
            # add data and labels to lists
            self.data_list.append(accel_data)
            self.labels_list.append(mapped_id)

    
    def parse_filename(self, filename):
        parts = filename.split('_')
        texture_id, line_degree, velocity, target_force, test_count = parts[0], parts[1], parts[2], parts[3], parts[4].split('.')[0]
        return texture_id, line_degree, velocity, target_force, test_count
    
    def load_csv(self, path):
        df = pd.read_csv(path)
        
        # generate target times
        start_time = df['time'].iloc[0]
        end_time = df['time'].iloc[-1]
        target_times = np.arange(start_time, end_time, self.sampling_interval)
        
        # find the nearest data point for each target time
        resampled_df = pd.DataFrame()
        for column in ['X', 'Y', 'Z']:
            nearest_indices = np.searchsorted(df['time'], target_times)
            nearest_indices = np.clip(nearest_indices, 0, len(df) - 1)
            resampled_df[column] = df[column].iloc[nearest_indices].values
        
        duration_samples = int(self.duration / self.sampling_interval)

        # zero mean adjustment
        resampled_df[["X", "Y", "Z"]] = resampled_df[["X", "Y", "Z"]] - resampled_df[["X", "Y", "Z"]].mean()

        # Last value padding
        if len(resampled_df) < duration_samples:
            padding_length = duration_samples - len(resampled_df)
            last_values = resampled_df.iloc[-1][["X", "Y", "Z"]]
            padding_df = pd.DataFrame({'X': [last_values["X"]]*padding_length, 
                                    'Y': [last_values["Y"]]*padding_length, 
                                    'Z': [last_values["Z"]]*padding_length})
            resampled_df = pd.concat([resampled_df, padding_df], ignore_index=True)

        return resampled_df[["X", "Y", "Z"]].values

    def _convert_labels_to_int(self, labels_list):
        unique_labels = sorted(set(labels_list))
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        return label_to_int
    
    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data = data.T  # shape: (T, C) -> (C, T)
        
        # add mel spectrogram conversion
        log_mel_data = self.compute_log_mel(data, sr=int(1/self.sampling_interval), n_fft=512, hop_length=128, n_mels=128)
        
        # rearrange dimensions
        log_mel_data = log_mel_data.transpose(1, 2, 0)

        if self.transform:
            log_mel_data = self.transform(log_mel_data)

        label = self.labels_list[idx]
        
        return log_mel_data, label

    # add mel spectrogram calculation function
    def compute_log_mel(self, data, sr=1000, n_fft=512, hop_length=256, n_mels=128):
        mel_spectrograms = []
        for i in range(data.shape[0]):
            S = np.abs(librosa.stft(data[i, :], n_fft=n_fft, hop_length=hop_length))**2
            mel_S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)
            log_mel_S = librosa.power_to_db(mel_S, ref=np.max)
            mel_spectrograms.append(log_mel_S)
        return np.array(mel_spectrograms)


# test
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((3, 1024))
    ])

    dataset = AccelVelDirDataset(0, '/workspace/texture_dataset', label="vel", transform=transform, duration=2)
    
    num_samples = len(dataset)
    print("Number of samples in dataset:", num_samples)
    print("Classes:", dataset.classes)

    # get random sample
    import random
    idx = random.randint(0, num_samples-1)

    # get random sample
    data, label = dataset[idx]
    print("Data shape:", data.shape)
    print("Label:", label)

    # plot data
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.title("accel data")
    # squeeze
    accel_data = data.squeeze(0)
    plt.plot(accel_data.T)

    plt.show()


