import os
import random
from PIL import Image
import tkinter as tk
from tkinter import filedialog

def select_folder(prompt):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(title=prompt)
    return folder_path

def crop_images(input_folder, output_folder, crop_size, num_crops):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            with Image.open(file_path) as img:
                # Get original DPI
                original_dpi = img.info.get('dpi', (72, 72))
                width, height = img.size
                crops = []
                
                for _ in range(num_crops):
                    if width < crop_size[0] or height < crop_size[1]:
                        # If the image is too small, skip cropping
                        print(f"Skipping {filename} as it is smaller than crop size")
                        continue
                    
                    left = random.randint(0, width - crop_size[0])
                    upper = random.randint(0, height - crop_size[1])
                    right = left + crop_size[0]
                    lower = upper + crop_size[1]
                    
                    crop = img.crop((left, upper, right, lower))
                    crops.append(crop)
                
                base_name, ext = os.path.splitext(filename)
                
                # Create a directory for each file's crops
                output_dir = os.path.join(output_folder, base_name)
                os.makedirs(output_dir, exist_ok=True)
                
                for i, crop in enumerate(crops):
                    output_filename = f"{base_name}_{i + 1}{ext}"
                    output_path = os.path.join(output_dir, output_filename)
                    # Save cropped image with original DPI
                    crop.save(output_path, dpi=original_dpi)

# Set the desired crop size and number of crops per image
crop_size = (224, 224)  # For example, 224x224 pixels
num_crops = 100  # Number of random crops per image

# Select the input and output folders
input_folder = select_folder("Select the input folder containing images")
output_folder = select_folder("Select the output folder for cropped images")

# Crop the images
crop_images(input_folder, output_folder, crop_size, num_crops)

print("Cropping completed.")