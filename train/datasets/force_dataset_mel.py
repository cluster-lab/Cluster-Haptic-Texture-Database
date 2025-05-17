import os
import pandas as pd
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                
from datasets.dataset_scale import DatasetScale

class ForceDatasetMel(Dataset):
    def __init__(self, root_dir, transform=None, sampling_interval=0.001, duration=4, scale=DatasetScale.FULL):
        self.force_dir = os.path.join(root_dir, "sensor_data/force")
        
        self.transform = transform
        self.sampling_interval = sampling_interval
        self.duration = duration
        
        folder_list = [os.path.join(self.force_dir, f) for f in os.listdir(self.force_dir) if os.path.isdir(os.path.join(self.force_dir, f))]
        folder_list = sorted(folder_list, key=lambda x: int(os.path.basename(x)))  # sort by texture id

        # scale
        if scale == DatasetScale.LITE:
            folder_list = folder_list[:DatasetScale.LITE.value]
        elif scale == DatasetScale.MEDIAN:
            folder_list = folder_list[:DatasetScale.MEDIAN.value]

        file_list = []
        for folder in folder_list:
            file_list.extend([f for f in os.listdir(folder) if f.endswith('.csv')])

        # get all texture ids
        all_texture_ids = [self.parse_filename(f)[0] for f in file_list]
        self.label_to_int = self._convert_labels_to_int(all_texture_ids)
        self.classes = list(self.label_to_int.keys())

        # list of data and labels
        self.data_list = []
        self.labels_list = []
        
        print("Loading ForceDataset...")
        total_files = len(file_list)
        for idx, file_name in enumerate(file_list):
            texture_id, _, _, _, _ = self.parse_filename(file_name)

            force_path = os.path.join(self.force_dir, texture_id, file_name)
            force_data = self.load_csv(force_path).astype(np.float32)  # データ型をfloat32に変換
            
            self.data_list.append(force_data)
            self.labels_list.append(self.label_to_int[texture_id])

            # print progress
            progress = (idx + 1) / total_files * 100
            print(f"Progress: {progress:.2f}% ({idx + 1}/{total_files})", end='\r')
        print("")
    
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
        # read force only
        nearest_indices = np.searchsorted(df['time'], target_times)
        # check array boundaries
        nearest_indices = np.clip(nearest_indices, 0, len(df) - 1)
        resampled_df['force'] = df['force'].iloc[nearest_indices].values
        
        duration_samples = int(self.duration / self.sampling_interval)

        # zero mean adjustment
        resampled_df['force'] = resampled_df['force'] - resampled_df['force'].mean()

        # Last value padding
        if len(resampled_df) < duration_samples:
            padding_length = duration_samples - len(resampled_df)
            last_value = resampled_df.iloc[-1]['force']
            padding_df = pd.DataFrame({'force': [last_value]*padding_length})
            resampled_df = pd.concat([resampled_df, padding_df], ignore_index=True)

        # return 1D data
        return resampled_df['force'].values.reshape(-1, 1)
    
    def _convert_labels_to_int(self, labels_list):
        unique_labels = sorted(set(labels_list))
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        return label_to_int
    
    def compute_log_mel(self, data, sr=1000, n_fft=512, hop_length=256, n_mels=128):
        # for 1D data
        S = np.abs(librosa.stft(data.squeeze(), n_fft=n_fft, hop_length=hop_length))**2
        mel_S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)
        log_mel_S = librosa.power_to_db(mel_S, ref=np.max)
        return log_mel_S[np.newaxis, ...]  # add channel dimension

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data = data.T  # shape: (T, C) -> (C, T)
        
        log_mel_data = self.compute_log_mel(data, sr=int(1/self.sampling_interval), n_fft=512, hop_length=128, n_mels=128)

        # (1, 2, 0)
        log_mel_data = log_mel_data.transpose(1, 2, 0)

        # apply transform
        if self.transform:
            log_mel_data = self.transform(log_mel_data)


        label = self.labels_list[idx]
        
        return log_mel_data, label
    

# for 1D data
def display_channels_side_by_side(log_mel_data):
    plt.figure(figsize=(10, 5))
    img = plt.imshow(log_mel_data.reshape(128, -1), aspect='auto', interpolation='none', origin='lower')
    plt.title('Force Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.colorbar(img, format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# test
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(),
        # transforms.Resize((224, 224)),
        transforms.Resize((128, 64)),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset = ForceDatasetMel(root_dir="/workspace/texture_dataset", transform=transform, sampling_interval=0.0002, duration=1, scale=DatasetScale.LITE)

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

    # for 1D data
    data = data.squeeze()  # remove extra dimensions
    display_channels_side_by_side(data)

    # transform transforms.RandomHorizontalFlip(),
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     #transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    #     #transforms.RandomErasing(p=0.5),
    # ])
    # data = transform(data)

    # display rgb spectrogram
    # squeeze
    # data = data.squeeze(0)
    # display_channels_side_by_side(data)

    # plot data
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(15,5))
    # plt.title("accel data")
    # # squeeze
    # accel_data = data.squeeze(0)
    # plt.imshow(accel_data, aspect='auto')
    # plt.colorbar()
    # plt.show()
