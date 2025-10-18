import os
import shutil

import cv2
import random
import numpy as np

from typing import List, Tuple


def is_black_frame(frame: np.ndarray, threshold: float) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)/255.0    # нормируем до [0,1]
    return mean_brightness < threshold

def split_files(file_list: List[str], train_ratio: float, val_ratio: float, n_samples: int) -> Tuple[List[str], List[str], List[str]]:
    if n_samples < len(file_list):
        file_list = random.sample(file_list, n_samples)

    random.shuffle(file_list)
    n_total = len(file_list)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train+n_val]
    test_files = file_list[n_train+n_val:]
    return train_files, val_files, test_files