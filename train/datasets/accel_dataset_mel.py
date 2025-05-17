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

class AccelDatasetMel(Dataset):
    def __init__(self, root_dir, transform=None, sampling_interval=0.001, duration=4, 
                 scale=DatasetScale.FULL, output_direction=False, output_velocity=False):
        self.accel_dir = os.path.join(root_dir, "sensor_data/accel")
        
        self.transform = transform
        self.sampling_interval = sampling_interval
        self.duration = duration
        self.output_direction = output_direction
        self.output_velocity = output_velocity
        
        folder_list = [os.path.join(self.accel_dir, f) for f in os.listdir(self.accel_dir) if os.path.isdir(os.path.join(self.accel_dir, f))]
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
        self.file_list = []
        
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
        
        print("Loading AccelDataset...")
        total_files = len(file_list)
        for idx, file_name in enumerate(file_list):
            texture_id, _, _, _, _ = self.parse_filename(file_name)

            accel_path = os.path.join(self.accel_dir, texture_id, file_name)
            accel_data = self.load_csv(accel_path).astype(np.float32)
            
            self.data_list.append(accel_data)
            self.labels_list.append(self.label_to_int[texture_id])
            self.file_list.append(file_name)

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
        for column in ['X', 'Y', 'Z']:
            nearest_indices = np.searchsorted(df['time'], target_times)
            # check array boundaries
            nearest_indices = np.clip(nearest_indices, 0, len(df) - 1)
            resampled_df[column] = df[column].iloc[nearest_indices].values
        
        duration_samples = int(self.duration / self.sampling_interval)

        # adjust data to zero mean
        resampled_df[["X", "Y", "Z"]] = resampled_df[["X", "Y", "Z"]] - resampled_df[["X", "Y", "Z"]].mean()

        # Last value padding
        if len(resampled_df) < duration_samples:
            padding_length = duration_samples - len(resampled_df)
            last_values = resampled_df.iloc[-1][["X", "Y", "Z"]]
            padding_df = pd.DataFrame({'X': [last_values["X"]]*padding_length, 
                                    'Y': [last_values["Y"]]*padding_length, 
                                    'Z': [last_values["Z"]]*padding_length})
            resampled_df = pd.concat([resampled_df, padding_df], ignore_index=True)

        # 外れ値の修正は現在コメントアウトされているので、同様にコメントアウトしたままにします
        # for column in ["X", "Y", "Z"]:
        #     z_scores = abs((resampled_df[column] - resampled_df[column].mean()) / resampled_df[column].std())
        #     threshold = 3 # 3 sigma（3標準偏差）を閾値として外れ値とみなす
        #     outliers = z_scores > threshold
        #     resampled_df.loc[outliers, column] = resampled_df[column].median()  # 外れ値を中央値で置き換え

        return resampled_df[["X", "Y", "Z"]].values
    
    def _convert_labels_to_int(self, labels_list):
        unique_labels = sorted(set(labels_list))
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        return label_to_int
    
    def compute_log_mel(self, data, sr=1000, n_fft=512, hop_length=256, n_mels=128):
        mel_spectrograms = []
        for i in range(data.shape[0]):  # dataのshapeは (C, T) と仮定
            S = np.abs(librosa.stft(data[i, :], n_fft=n_fft, hop_length=hop_length))**2
            mel_S = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)
            log_mel_S = librosa.power_to_db(mel_S, ref=np.max)
            mel_spectrograms.append(log_mel_S)
        return np.array(mel_spectrograms)

    def normalize_value(self, value, min_val, max_val):
        """normalize value to 0-1 range"""
        if min_val == max_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data = data.T  # shape: (T, C) -> (C, T)
        
        log_mel_data = self.compute_log_mel(data, sr=int(1/self.sampling_interval), n_fft=512, hop_length=128, n_mels=128)
        log_mel_data = log_mel_data.transpose(1, 2, 0)

        # apply transform
        if self.transform:
            log_mel_data = self.transform(log_mel_data)

        label = self.labels_list[idx]

        # prepare basic outputs
        outputs = [log_mel_data, label]

        # process direction and velocity if needed
        if self.output_direction or self.output_velocity:
            filename = self.file_list[idx]
            _, direction, velocity, _, _ = self.parse_filename(filename)
            
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

# display channels side by side
def display_channels_side_by_side(log_mel_data):
    num_channels = log_mel_data.shape[0]
    fig, axes = plt.subplots(1, num_channels, figsize=(20, 5))

    for i in range(num_channels):
        img = axes[i].imshow(log_mel_data[i], aspect='auto', interpolation='none', origin='lower')
        axes[i].set_title(f'Channel {i + 1}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Mel Frequency')
        fig.colorbar(img, ax=axes[i], format='%+2.0f dB')

    plt.tight_layout()
    plt.show()
    

# test
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(),
        # transforms.Resize((224, 224)),
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    dataset = AccelDatasetMel(root_dir="/workspace/texture_dataset", transform=transform, sampling_interval=0.0002, duration=1, scale=DatasetScale.LITE)

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

    # display rgb spectrogram
    # squeeze
    data = data.squeeze(0)
    display_rgb_spectrogram(data)
    display_channels_side_by_side(data)

