# Dataset details

## Dataset URL
https://drive.google.com/drive/folders/1mNrVh7y1hK1eHlkC_qke7acv578rkf6D

## Directory structure
```
texture_dataset/
│
├── images
    │
    ├── images_scan_area/ (images of 100x100mm scan area)
    │
    └── crop_images/ (for evaluation of texture classification)
│
└── sensor_data/
    ├── audio/ (after noise canceling)
    │
    ├── raw_audio/
    │
    ├── accel/
    │
    ├── force/
    │
    └── position/
```

## images_scan_area file naming rule
- img(texture_id).jpg
- example: img0.jpg


## sensor_data file naming rule
- (texture_id)\_(direction)\_(velocity)\_(force)\_(repeat count).csv or .wav

### texture_id
[texture list](https://docs.google.com/spreadsheets/d/1ZQ6jQgE8iEozCCI4KmeYy866WPxxT0qG2Sy3mrofjE4/edit?usp=sharing)

### direction
clockwise angle(degree)  
(0, 45, 90, 135, 180, 225, 270, 315 degree)

### velocity
scan speed (mm/s)  
(20, 30, 40, 50, 60 mm/s)

### force
push force (mN)  
(500 or 1000 mN)

### repeat count
numbers of scan  
(0 or 1)

### example: 0_135_40_500_0.csv  
- texture id: 0
- direction 135 degree
- velocity 40 mm/s
- force 500 mN
- repeat count: 0

## audio
Sound source after noise cancelling.
- file format : wav
- sampling rate 44.1 kHz

## raw_audio
Raw sound source
- file format : wav
- two channel (stereo signal)
    - channnel 0: main mic signal (texture + machine noise signal)
    - channnel 1: sub mic signal (machine noise signal)
- sampling rate 44.1 kHz


## accel
- file format : csv
- time (s)
- 3 axis accelaration (g)
- sampling rate 6 kHz
### csv format
```
time,X,Y,Z
0.0009990528727670048,0.014,0.067,-0.96
0.0012090528727670066,0.034,0.048,-0.946
0.0013390528727670048,0.029,0.062,-0.952
```

## force 
- file format : csv
- time (s)
- force (N)
- sampling rate 6 kHz
### csv format
```
time,force
0.0009990528727670048,1.009
0.0012090528727670066,1.009
0.0013390528727670048,1.009
```

## position
- file format : csv
- time (s)
- 2D position (mm)
- sampling rate 100 Hz
### csv format
```
time,X,Y
0.000613,0.0,45.0
0.010477,0.0,45.0
0.01994,0.0,45.0
```

