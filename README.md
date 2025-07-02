# Cluster Haptic Texture Database: Haptic Texture Database with Varied Velocity–Direction Sliding Contacts

The Cluster Haptic Texture Database is a comprehensive dataset for haptic texture research. This database provides haptic texture data with various sliding velocities and directions.
This repository contains the dataset and experiment code.

- [Paper](https://arxiv.org/abs/2407.16206)

![](documents/reprensentative.jpg)


## Dataset

[Dataset and detail](documents/dataset_details.md)

## Installation
### 1. git clone
```
git clone git@github.com:cluster-lab/Cluster-Haptic-Texture-Database.git
```
### 2. build docker image
```
cd Cluster-Haptic-Texture-Database
./docker_build.sh
```
### 3. unzip [dataset](https://doi.org/10.6084/m9.figshare.29438288) and place in this directory
directory structure
```
Cluster-Haptic-Texture-Database
|
|--texture_dataset
   |
   |--images
   |
   |--sensor_data
```

### 4. make container
```
./docker_run.sh
```
### (option: venv ver)
```
python3 -m venv .venv

# windows
.venv\Scripts\activate.ps1

# install pip package
pip3 install -r requirements.txt
```

## Data Viewer
```
python3 viewer/AudioViewer.py
python3 viewer/AccelerationViewer.py
python3 FFT_view.py
```

## Classification experiments
**Options**
- `--models`: Specify the models to use (resnet, vit, svc, decision_tree, random_forest)
- `--dataset`: Specify the dataset type (audio, accel, audio_accel)
- `--scale`: Specify the dataset scale (lite, medium, full)
- `--labels`: Specify the classification labels (velocity, direction)
### image texture classification
```
python3 train/image_classification.py --models resnet
```
### haptics texture classification
```
python3 train/haptics_texture_classification.py --dataset audio --models resnet --scale full
```
### haptics texture classification with motion label
```
python3 train/haptics_texture_classification_motion.py --dataset audio --models resnet --scale full
```
### haptics velocity and direction classification
```
python3 train/vel_dir_classification.py --dataset audio --labels velocity --model resnet
```

### log
```
tensorboard --logdir=.log
```

## Citing
```
@misc{eguchi2024clusterhaptictexturedatabase,
      title={Cluster Haptic Texture Database: Haptic Texture Database with Variety in Velocity and Direction of Sliding Contacts}, 
      author={Michikuni Eguchi and Madoka Ito and Tomohiro Hayase and Yuichi Hiroi and Takefumi Hiraki},
      year={2024},
      eprint={2407.16206},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2407.16206}, 
}
```