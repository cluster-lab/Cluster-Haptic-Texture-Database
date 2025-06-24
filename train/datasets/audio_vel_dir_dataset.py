import os
import numpy as np
import librosa
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                
from datasets.dataset_scale import DatasetScale


class AudioVelDirDataset(Dataset):
    def __init__(self, index, root_dir, label="vel", transform=None, duration=4, sr=22050):
        self.audio_dir = os.path.join(root_dir, "sensor_data/audio")

        self.transform = transform

        self.duration = duration  # seconds
        self.sr = sr
        self.total_samples = sr * duration
        
        folder_list = [os.path.join(self.audio_dir, f) for f in os.listdir(self.audio_dir) if os.path.isdir(os.path.join(self.audio_dir, f))]
        folder_list = sorted(folder_list, key=lambda x: int(os.path.basename(x)))  # sort by texture id
        file_list = []
        for folder in folder_list:
            file_list.extend([f for f in os.listdir(folder) if f.endswith('.wav')])

        # get all texture_ids
        all_texture_ids = sorted([int(self.parse_filename(f)[0]) for f in file_list])
        # create a list of unique texture_ids
        unique_texture_ids = [str(id) for id in sorted(set(all_texture_ids))]
        # get the texture_id at the specified index
        target_texture_id = unique_texture_ids[index]
        # get data of the specified texture_id only
        self.file_list = [f for f in file_list if self.parse_filename(f)[0] == target_texture_id]
        self.file_list = sorted(self.file_list)


        # get all feedrates
        if label == "vel":
            all_label = [self.parse_filename(f)[2] for f in self.file_list]
        elif label == "dir":
            all_label = [self.parse_filename(f)[1] for f in self.file_list]
        self.label_to_int = self._convert_labels_to_int(all_label)
        self.classes = list(self.label_to_int.keys())

        # save data and label
        self.data_list = []
        self.labels_list = []
        for file_name in self.file_list:
            texture_id, line_degree, velocity, _, _ = self.parse_filename(file_name)

            audio_path = os.path.join(self.audio_dir, texture_id, file_name)
            audio_data = self.load_wav(audio_path)

            if label == "vel":
                mapped_id = self.label_to_int[velocity]
            elif label == "dir":
                mapped_id = self.label_to_int[line_degree]
            
            # add data and label to lists
            self.data_list.append(audio_data)
            self.labels_list.append(mapped_id)
        

    
    def parse_filename(self, filename):
        parts = filename.split('_')
        texture_id, line_degree, velocity, target_force, test_count = parts[0], parts[1], parts[2], parts[3], parts[4].split('.')[0]
        return texture_id, line_degree, velocity, target_force, test_count
    
    def load_wav(self, path):
        waveform, sr = librosa.load(path, sr=self.sr)

        if len(waveform) < self.total_samples:
            waveform = np.pad(waveform, (0, self.total_samples - len(waveform)))

        feature_melspec = librosa.feature.melspectrogram(y=waveform[:self.total_samples], sr=sr)
        feature_melspec_db = librosa.power_to_db(feature_melspec, ref=np.max)

        return feature_melspec_db
    
    def _convert_labels_to_int(self, labels_list):
        unique_labels = sorted(set(labels_list))
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        return label_to_int
    
    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        # apply transform
        if self.transform:
            data = self.transform(data)

        label = self.labels_list[idx]

        return data, label


# test
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 256))
    ])

    dataset = AudioVelDirDataset(25, '/workspace/texture_dataset', label="vel", transform=transform, duration=4, sr=22050)
    
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
    plt.figure(figsize=(15,5))
    plt.title("mel spectrogram")
    # squeeze
    data = data.squeeze(0)
    plt.imshow(data, aspect='auto', origin='lower')
    plt.colorbar()
    plt.show()