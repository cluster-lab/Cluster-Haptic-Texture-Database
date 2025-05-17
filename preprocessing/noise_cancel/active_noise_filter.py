import os
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from active_filter.LMSnoise_cancel import LMSNoiseCanceller
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

def apply_filter_and_save(input_file, input_dir, output_dir):

    canceller = LMSNoiseCanceller(input_file, 
                                  filter_length=700, 
                                  mu=1.0,
                                  algorithm='nlms',  # Use normalized LMS
                                  leakage=0.001,     # Set leakage coefficient
                                  prewhiten=False,    # Enable prewhitening
                                  filter_mode='noncausal')    # Use causal filter
    
    filtered_data = canceller.run_lms()

    # determine the output path
    relative_path = os.path.relpath(input_file, input_dir)
    output_file = os.path.join(output_dir, relative_path)
    output_folder = os.path.dirname(output_file)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    canceller.save_error_wav(output_file)

def process_wav_files(input_dir, output_dir, filter_class, scale=None):
    wav_files = read_wav_files_from_folder(input_dir, scale)
    for wav_file in tqdm(wav_files, desc="Processing WAV files"):
        # apply filter and save the result
        apply_filter_and_save(wav_file, input_dir, output_dir)

# example
if __name__ == '__main__':
    input_directory = '/workspace/texture_dataset/sensor_data/raw_sound'
    output_directory = '/workspace/texture_dataset/sensor_data/sound'

    # if not exists, create
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    process_wav_files(input_directory, output_directory, LMSNoiseCanceller)
    