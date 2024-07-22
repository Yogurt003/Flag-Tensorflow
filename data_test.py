import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

def prepare_images_from_folder(base_folder_path):
    labeled_image_arrays = []

    for root, dirs, files in os.walk(base_folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                label = os.path.basename(root)
                file_path = os.path.join(root, file)

                pil_image = Image.open(file_path).convert("L")
                pil_image = pil_image.resize((150, 150))
                pil_image = ImageOps.invert(pil_image)

                image_array = np.array(pil_image)
                image_array = image_array.flatten()

                labeled_image_arrays.append( image_array.tolist())

    return labeled_image_arrays

def save_images_to_csv(labeled_image_arrays, csv_path):
    column_names = [f'pixel{i}' for i in range(150 * 150)]
    df = pd.DataFrame(labeled_image_arrays, columns=column_names)

    df.to_csv(csv_path, index=False, float_format='%d')

base_folder_path = 'D:\\barovinh\\Python\\Dataset\\test2024\\test2024'
csv_path = 'D:\\barovinh\\Python\\Dataset\\test.csv'

labeled_image_arrays = prepare_images_from_folder(base_folder_path)
save_images_to_csv(labeled_image_arrays, csv_path)
