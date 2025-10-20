import os
import cv2
import random
import shutil
import zipfile
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

from utils import is_black_frame, split_files


FOLDER_RAW = "data/raw_data/"     # путь до папки с архивом 
FOLDER_FRAMES = "data/frames/"    # путь до папки с фреймами
IMAGE_SIZE = (224, 224)           # нужный размер изображений
BRIGHTNESS_THRESHOLD = 0.1       #? порог яркости, чтобы обрезать черные фреймы
TRAIN_RATIO = 0.7
SEED = 42

random.seed(SEED)


def main():
    with zipfile.ZipFile(f"{FOLDER_RAW}5952179.zip", 'r') as zip_ref:    #? можно добавить загрузку с облака
        zip_ref.extractall(FOLDER_RAW)

    for subset in ["train", "test"]:    # структура папок для фреймов
        subset_path = os.path.join(FOLDER_FRAMES, subset)
        os.makedirs(subset_path, exist_ok=True)

    for i in tqdm(range(1, 8)):
        cap = cv2.VideoCapture(FOLDER_RAW+f"jp2c00948_si_00{i}.mp4")

        class_name = f"class_{i}"
        frames_list = []

        index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break 

            if is_black_frame(frame, threshold=BRIGHTNESS_THRESHOLD):    # проверка на черные фреймы
                continue 

            resized_frame = cv2.resize(frame, IMAGE_SIZE)    # изменение размера фрейма
            frames_list.append(resized_frame)
            index += 1

        cap.release()

        train_files, test_files = split_files(frames_list, train_ratio=TRAIN_RATIO, n_samples=None)    # разделение

        for subset_name, subset_files in zip(["train", "test"], [train_files,  test_files]):
            subset_class_dir = os.path.join(FOLDER_FRAMES, subset_name, class_name)
            os.makedirs(subset_class_dir, exist_ok=True)
            for j, frame in enumerate(subset_files):
                cv2.imwrite(os.path.join(subset_class_dir, f"frame{j}.jpg"), frame)


if __name__ == "__main__":
    main()
    