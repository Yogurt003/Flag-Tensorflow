import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

label_map = {
    'coi-82': 0,
    'co-sch-bchqs-quan-huyen': 1,
    'co-vi-tri-chi-huy-dai-doi': 2,
    'co-vi-tri-chi-huy-tieu-doan': 3,
    'co-vi-tri-chi-huy-trung-doi': 4,
    'cum-xe-thiet-giap': 5,
    'dai-doi-coi-106,7': 6,
    'danh-pha': 7,
    'diem-tua-phong-ngu': 8,
    'hang-rao-thep-gai': 9,
    'khu-vuc-do-bo-duong-khong': 10,
    'ki-hieu-xe-tang': 11,
    'may-bay-truc-thang-vu-trang': 12,
    'sung-DKZ-co-75': 13,
    'tran-dia-trung-doi-coi-82': 14
}

def prepare_images_from_folder(base_folder_path):
    labeled_image_arrays = []

    for root, dirs, files in os.walk(base_folder_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                label = os.path.basename(root)
                numerical_label = label_map.get(label, 0)
                file_path = os.path.join(root, file)

                pil_image = Image.open(file_path).convert("L")
                pil_image = pil_image.resize((150, 150))
                pil_image = ImageOps.invert(pil_image)

                image_array = np.array(pil_image)
                image_array = image_array.flatten()

                labeled_image_arrays.append([numerical_label] + image_array.tolist())

    return labeled_image_arrays

def save_images_to_csv(labeled_image_arrays, csv_path):
    column_names = ['label'] + [f'pixel{i}' for i in range(150 * 150)]
    df = pd.DataFrame(labeled_image_arrays, columns=column_names)

    df.to_csv(csv_path, index=False, float_format='%d')

base_folder_path = 'D:\\barovinh\\Python\\Dataset\\train2024\\train2024'
csv_path = 'D:\\barovinh\\Python\\Dataset\\train.csv'

labeled_image_arrays = prepare_images_from_folder(base_folder_path)
save_images_to_csv(labeled_image_arrays, csv_path)
