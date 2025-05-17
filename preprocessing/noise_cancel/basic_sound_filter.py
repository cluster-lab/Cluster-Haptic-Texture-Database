import os
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from basic_filter.HighPassFilter import HighPassFilter
from basic_filter.LowPassFilter import LowPassFilter
from basic_filter.BandPassFilter import BandPassFilter
import re

def read_wav_files_from_folder(sound_dir, scale=None):
    folder_list = [os.path.join(sound_dir, f) for f in os.listdir(sound_dir) if os.path.isdir(os.path.join(sound_dir, f))]
    folder_list = sorted(folder_list, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))  # sort by texture id

    # scale
    if scale:
        folder_list = folder_list[:scale]

    file_list = []
    for folder in folder_list:
        file_list.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')])
    
    return file_list

def apply_filter_and_save(input_file, input_dir, output_dir, filter_class, filter_args):
    fs, data = wavfile.read(input_file)

    filter_instance = filter_class(*filter_args)
    filtered_data = filter_instance.apply(data)

    # determine the output path
    relative_path = os.path.relpath(input_file, input_dir)
    output_file = os.path.join(output_dir, relative_path)
    output_folder = os.path.dirname(output_file)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    wavfile.write(output_file, fs, filtered_data.astype(np.int16))

def process_wav_files(input_dir, output_dir, filter_class, filter_args, scale=None):
    wav_files = read_wav_files_from_folder(input_dir, scale)
    for wav_file in tqdm(wav_files, desc="Processing WAV files"):
        apply_filter_and_save(wav_file, input_dir, output_dir, filter_class, filter_args)

# example
if __name__ == '__main__':
    input_directory = 'C:/workspace/hapticsdataset_eval/texture_dataset/sensor_data/sound'
    output_directory = 'C:/workspace/hapticsdataset_eval/texture_dataset/sensor_data/filter_sound'
    # input_directory = '/workspace/texture_dataset/sensor_data/sound'
    # output_directory = '/workspace/texture_dataset/sensor_data/filter_sound'
    cutoff_frequency = 1000  # cutoff frequency of high-pass filter
    sampling_rate = 44100  # sampling rate
    order = 12

    lowcut_frequency = 1000
    highcut_frequency = 5000

    process_wav_files(input_directory, output_directory, HighPassFilter, (cutoff_frequency, sampling_rate, order))
    # process_wav_files(input_directory, output_directory, LowPassFilter, (cutoff_frequency, sampling_rate, order))
    # process_wav_files(input_directory, output_directory, BandPassFilter, (lowcut_frequency, highcut_frequency, sampling_rate, 5))