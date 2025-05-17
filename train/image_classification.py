import os
import time
import copy
import argparse
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from classification.ResnetClassification import ResnetClassification
from classification.VitClassification import VitClassification


def get_data_transforms(dataset_type):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    if dataset_type == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    elif dataset_type == 'val' or dataset_type == 'test':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def main(args):
    # data preprocessing
    data_transforms = {
        'train': get_data_transforms('train'),
        'val': get_data_transforms('val'),
        'test': get_data_transforms('test')
    }

    # load dataset
    full_dataset = datasets.ImageFolder('texture_dataset/images/crop_images', transform=data_transforms['train'])

    # define dataset size
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # split dataset
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # redefine dataloader
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    }

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
    class_names = full_dataset.classes

    if 'resnet' in args.models:
        net = ResnetClassification(
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            class_names=class_names,
            experiment_name='image_classification_resnet',
            num_epochs=50,
            learning_rate=0.0001
        )
    elif 'vit' in args.models:
        net = VitClassification(
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            class_names=class_names,
            experiment_name='image_classification_vit',
            num_epochs=50,
            learning_rate=0.0001
        )

    net.open_writer()
    net.train_model()
    net.save_model()
    net.evaluate_model()
    net.close_writer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Haptics Texture Classification')
    parser.add_argument('--models', nargs='+', choices=['resnet', 'vit'], required=True,
                        help='Models to use for classification (resnet, vit)')
    args = parser.parse_args()

    main(args)