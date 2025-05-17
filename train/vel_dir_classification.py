import os
import time
import copy
import argparse
import torch
import torchvision
from datetime import datetime
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
import numpy as np
from torchmetrics.classification import MulticlassConfusionMatrix
from datasets.sound_vel_dir_dataset import SoundVelDirDataset
from datasets.accel_vel_dir_dataset import AccelVelDirDataset
from datasets.sound_accel_vel_dir_dataset import SoundAccelVelDirDataset
from classification.ResnetClassification import ResnetClassification
from classification.VitClassification import VitClassification
from classification.SoundAccelMultiClassification import SoundAccelMultiClassification



def plot_accuracy_bar(acc_list, title, texture_num):

    if len(np.array(acc_list).shape) > 1 and np.array(acc_list).shape[1] > 1:
        means = np.mean(acc_list, axis=0)
        variances = np.var(acc_list, axis=0)
        y_values = means
        y_err = variances
    else:
        y_values = acc_list
        y_err = None

    plt.figure(figsize=(20, 10))
    plt.rcParams["font.size"] = 28
    cap_value = 10 if y_err is not None else 0
    plt.bar(range(texture_num), y_values, yerr=y_err, capsize=cap_value)

    # draw y=1.0 line
    plt.axhline(y=1.0, color='r', linestyle='--')

    # Set y-axis limits
    plt.ylim(0, 1.1)  # <-- This line sets the y-axis limits


    # set x-axis ticks
    plt.xticks(range(0, texture_num, 10))
    plt.gca().set_xticks(range(texture_num), minor=True)

    # set minor tick
    plt.gca().tick_params(axis='x', which='minor', length=5, width=2)

    # set major tick
    plt.gca().tick_params(axis='x', which='major', length=10, width=4)

    plt.xlabel("texture id")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.show()

def log_csv(log_dir, experiment_name, acc_list, texture_num):
    # if directory does not exist, create it
    os.makedirs(log_dir, exist_ok=True)
    
    # Save the results to a CSV file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_filename = f"{log_dir}/{experiment_name}_{timestamp}.csv"
    with open(csv_filename, 'w') as f:
        f.write("texture_id,accuracy\n")
        for texture_id, acc in enumerate(acc_list):
            f.write(f"{texture_id},{acc}\n")
    print(f"Results saved to {csv_filename}")

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

    # define additional transforms for training
    train_augmentations = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.RandomErasing(p=0.2)
    ]

    # define additional resize for vit model
    if 'vit' in model_type:
        base_transforms.insert(2, transforms.Resize((224, 224)))

    # assemble transforms based on dataset type
    if dataset_type == 'train':
        transforms_list = base_transforms + train_augmentations
    elif dataset_type in ['val', 'test']:
        transforms_list = base_transforms
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return transforms.Compose(transforms_list)
    
def load_dataset(dataset_type, texture_id, data_dir, label, transform):
    if dataset_type == 'sound':
        return SoundVelDirDataset(texture_id, data_dir, label=label, transform=transform, duration=1, sr=22050)
    elif dataset_type == 'accel':
        return AccelVelDirDataset(texture_id, data_dir, label=label, transform=transform, sampling_interval=0.0002, duration=1)
    elif dataset_type == 'sound_accel':
        return SoundAccelVelDirDataset(texture_id, data_dir, label=label, transform=transform, duration=1, sr=22050, sampling_interval=0.0002)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def dataset_prepare(texture_id, root_directory, label, args):
    # data preprocessing
    data_transforms = {
        'train': get_data_transforms('train', args.model),
        'val': get_data_transforms('val', args.model),
        'test': get_data_transforms('test', args.model),
    }

    # load dataset
    full_dataset = load_dataset(args.dataset, texture_id, root_directory, label, data_transforms['train'])

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

    return dataloaders, dataset_sizes, class_names

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(args, dataloaders, dataset_sizes, class_names, experiment_name):
    if args.dataset == 'sound' or args.dataset == 'accel':
        if args.model == 'resnet':
            return ResnetClassification(
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                class_names=class_names,
                experiment_name=experiment_name,
                num_epochs=30,
                learning_rate=0.0001
            )
        elif args.model == 'vit':
            return VitClassification(
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                class_names=class_names,
                experiment_name=experiment_name,
                num_epochs=30,
                learning_rate=0.0001
            )
        
    elif args.dataset == 'sound_accel':
        if args.model == 'resnet':
            return SoundAccelMultiClassification(
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                class_names=class_names,
                experiment_name=experiment_name,
                num_epochs=30,
                learning_rate=0.0001,
                model="resnet",
                feature_size=128
            )
        elif args.model == 'vit':
            return SoundAccelMultiClassification(
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                class_names=class_names,
                experiment_name=experiment_name,
                num_epochs=30,
                learning_rate=0.0001,
                model="vit",
                feature_size=128
            )
        

def main(args):
    # if seed is set, apply it
    if args.seed is not None:
        set_seed(args.seed)

    dataset_dir = '/workspace/texture_dataset'
    texture_num = 118
    log_dir = '/workspace/.log/vel_dir_classification'

    if 'velocity' in args.labels:
        experimet_name = "velocity_classification_" + args.dataset + "_" + args.model

        print("Velocity classification Start!!!")
        vel_acc_list = []
        for texture_id in range(texture_num):
            print("--------------------------------")
            print(f"Texture ID: {texture_id}")
            dataloaders, dataset_sizes, class_names = dataset_prepare(texture_id, dataset_dir, 'vel', args)
            
            # get model
            model = get_model(args, dataloaders, dataset_sizes, class_names, experimet_name)
            model.train_model(log_visualization=False)
            vel_acc = model.evaluate_model(log_visualization=False)
            vel_acc_list.append(vel_acc)
            print(f"Test Accuracy: {vel_acc}")
        
        log_csv(log_dir, experimet_name, vel_acc_list, texture_num)
        plot_accuracy_bar(vel_acc_list, f"Velocity Classification Accuracy ({args.model})", texture_num)

    if 'direction' in args.labels:
        experimet_name = "direction_classification_" + args.dataset + "_" + args.model

        print("Direction classification Start!!!")
        dir_acc_list = []
        for texture_id in range(texture_num):
            print("--------------------------------")
            print(f"Texture ID: {texture_id}")
            dataloaders, dataset_sizes, class_names = dataset_prepare(texture_id, dataset_dir, 'dir', args)
            
            # get model
            model = get_model(args, dataloaders, dataset_sizes, class_names, experimet_name)
            model.train_model(log_visualization=False)
            dir_acc = model.evaluate_model(log_visualization=False)
            dir_acc_list.append(dir_acc)
            print(f"Test Accuracy: {dir_acc}")

        log_csv(log_dir, experimet_name, dir_acc_list, texture_num)
        plot_accuracy_bar(dir_acc_list, f"Direction Classification Accuracy ({args.model})", texture_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Haptics Texture Classification')
    parser.add_argument('--dataset', type=str, choices=['sound', 'accel', 'sound_accel'], required=True,
                        help='Dataset to use for classification (sound, accel, sound_accel)')
    parser.add_argument('--labels', nargs='+', choices=['velocity', 'direction'], required=True,
                        help='Labels to classify (velocity, direction)')
    parser.add_argument('--model', type=str, choices=['resnet', 'vit'], required=True,
                        help='Model to use for classification (resnet, vit)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    main(args)