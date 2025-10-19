import os
import shutil

import cv2
import random
import numpy as np

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from typing import List, Tuple


def is_black_frame(frame: np.ndarray, threshold: float) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)/255.0    # нормируем до [0,1]
    return mean_brightness < threshold

def split_files(file_list: List[str], train_ratio: float, val_ratio: float, 
                n_samples: int) -> Tuple[List[str], List[str], List[str]]:
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

def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
                optimizer: Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
