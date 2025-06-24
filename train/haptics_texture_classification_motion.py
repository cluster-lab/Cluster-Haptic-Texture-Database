import os
import time
import copy
import argparse
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
import numpy as np
from torchmetrics.classification import MulticlassConfusionMatrix
from datasets.audio_dataset import AudioDataset
from datasets.accel_dataset_mel import AccelDatasetMel
from datasets.force_dataset_mel import ForceDatasetMel
from datasets.audio_accel_dataset import AudioAccelDataset
from datasets.dataset_scale import DatasetScale
from classification.MotionTokenClassification import MotionTokenClassification

class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None, multi_input=False):
        self.dataset = dataset
        self.transform = transform
        self.multi_input = multi_input

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.multi_input:
            sample1, sample2, label, direction, velocity = self.dataset[idx]
            if self.transform:
                sample1 = self.transform(sample1)
                sample2 = self.transform(sample2)
            return sample1, sample2, label, direction, velocity
        else:
            sample, label, direction, velocity = self.dataset[idx]
            if self.transform:
                sample = self.transform(sample)
            return sample, label, direction, velocity


def get_data_transforms(dataset_type, model_type):
    # set mean and std
    mean = [0.5]
    std = [0.5]

    # define basic transforms
    base_transforms = [
        transforms.ToTensor(),
        transforms.Resize((128, 64)),
        transforms.Normalize(mean=mean, std=std)
    ]

    # return basic transforms for traditional machine learning models
    if any(model in model_type for model in ['svc', 'decision_tree', 'random_forest']):
        return transforms.Compose(base_transforms)

    # additional transforms for training
    train_augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomErasing(p=0.2)
    ]

    # additional resize for VIT model
    if 'vit' in model_type:
        base_transforms.insert(2, transforms.Resize((224, 224)))

    # assemble transforms according to dataset type
    if dataset_type == 'train':
        transforms_list = base_transforms + train_augmentations
    elif dataset_type in ['val', 'test']:
        transforms_list = base_transforms
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return transforms.Compose(transforms_list)


def load_dataset(dataset_type, dataset_scale, data_dir, transform):
    # select scale
    if dataset_scale == 'full':
        scale = DatasetScale.FULL
    elif dataset_scale == 'medium':
        scale = DatasetScale.MEDIAN
    elif dataset_scale == 'lite':
        scale = DatasetScale.LITE
    else:
        raise ValueError(f"Unknown dataset scale: {dataset_scale}")

    if dataset_type == 'audio':
        return AudioDataset(data_dir, transform=transform, duration=1, sr=22050, scale=scale, output_direction=True, output_velocity=True)
    elif dataset_type == 'accel':
        return AccelDatasetMel(data_dir, transform=transform, sampling_interval=0.0002, duration=1, scale=scale, output_direction=True, output_velocity=True)
    elif dataset_type == 'audio_accel':
        return AudioAccelDataset(data_dir, transform=transform, duration=1, sr=22050, sampling_interval=0.0002, scale=scale, output_direction=True, output_velocity=True)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    if args.seed is not None:
        set_seed(args.seed)
    
    # data preprocessing
    data_transforms = {
        'train': get_data_transforms('train', args.models),
        'val': get_data_transforms('val', args.models),
        'test': get_data_transforms('test', args.models)
    }

    # load dataset
    full_dataset = load_dataset(args.dataset, args.scale, '/workspace/texture_dataset', None)

    # define dataset size
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # split dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # redefine dataset
    if args.dataset == 'audio_accel':
        train_dataset = CustomDataset(train_dataset, transform=data_transforms['train'], multi_input=True)
        val_dataset = CustomDataset(val_dataset, transform=data_transforms['val'], multi_input=True)
        test_dataset = CustomDataset(test_dataset, transform=data_transforms['test'], multi_input=True)
    else:
        train_dataset = CustomDataset(train_dataset, transform=data_transforms['train'])
        val_dataset = CustomDataset(val_dataset, transform=data_transforms['val'])
        test_dataset = CustomDataset(test_dataset, transform=data_transforms['test'])

    # redefine dataloader
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    }

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
    class_names = full_dataset.classes

    if 'vit' in args.models:
        if args.dataset == 'audio':
            # audio
            print("Start audio vit")
            net = MotionTokenClassification(
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                class_names=class_names,
                feature_size=256,
                experiment_name='motion_token_audio_texture_classification_vit',
                num_epochs=50,
                learning_rate=0.0001,
                model="vit",
                use_audio=True,
                use_accel=False
            )
        elif args.dataset == 'accel':
            # accel
            print("Start accel vit")
            net = MotionTokenClassification(
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                class_names=class_names,
                feature_size=256,
                experiment_name='motion_token_accel_texture_classification_vit',
                num_epochs=50,
                learning_rate=0.0001,
                model="vit",
                use_audio=False,
                use_accel=True
            )
        else:
            # audio+accel
            print("Start audio+accel vit")
            net = MotionTokenClassification(
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                class_names=class_names,
                feature_size=256,
                experiment_name='motion_token_audio_accel_texture_classification_vit',
                num_epochs=50,
                learning_rate=0.0001,
                model="vit",
                use_audio=True,
                use_accel=True
            )

        net.open_writer()
        net.train_model()
        net.save_model()
        net.evaluate_model()
        net.close_writer()

    if 'resnet' in args.models:
        if args.dataset == 'audio':
            # audio
            print("Start audio resnet")
            net = MotionTokenClassification(
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                class_names=class_names,
                feature_size=256,
                experiment_name='motion_token_audio_texture_classification_resnet',
                num_epochs=50,
                learning_rate=0.0001,
                model="resnet",
                use_audio=True,
                use_accel=False
            )
        elif args.dataset == 'accel':
            print("Start accel resnet")
            # accel
            net = MotionTokenClassification(
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                class_names=class_names,
                feature_size=256,
                experiment_name='motion_token_accel_texture_classification_resnet',
                num_epochs=50,
                learning_rate=0.0001,
                model="resnet",
                use_audio=False,
                use_accel=True
            )
        elif args.dataset == 'audio_accel':
            # audio+accel
            print("Start audio+accel resnet")
            net = MotionTokenClassification(
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                class_names=class_names,
                feature_size=256,
                experiment_name='motion_token_audio_accel_texture_classification_resnet',
                num_epochs=50,
                learning_rate=0.0001,
                model="resnet",
                use_audio=True,
                use_accel=True
            )
        
        net.open_writer()
        net.train_model()
        net.save_model()
        net.evaluate_model()
        net.close_writer()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Haptics Texture Classification')
    parser.add_argument('--dataset', type=str, choices=['audio', 'accel', 'audio_accel'], required=True,
                        help='Dataset to use for classification (audio, accel, audio_accel)')
    parser.add_argument('--models', nargs='+', choices=['vit', 'resnet'], required=True,
                        help='Models to use for classification (vit, resnet)')
    parser.add_argument('--scale', type=str, choices=['full', 'medium', 'lite'], default='full',
                        help='Dataset scale (full, medium, lite)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    main(args)