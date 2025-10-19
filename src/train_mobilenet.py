from tqdm import tqdm

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import train_epoch, evaluate


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCH = 10


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # mean и std ImageNet
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder('data/frames/train', transform=transform)
    test_dataset = datasets.ImageFolder('data/frames/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256)

    model = models.mobilenet_v3_small(pretrained=True)
    # print(model)

    for name, param in model.named_parameters():
        # print(name)
        if ("12" in name) | ("11" in name) | ("10" in name):
            param.requires_grad = True   
        else: 
            param.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 7)    # Изменение на 7 классов

    model = model.to(device)
    print(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    for _ in tqdm(range(NUM_EPOCH)):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # val_acc = evaluate(model, val_loader, device)
    
    test_acc, test_report = evaluate(model, test_loader, device)
    print(test_report)

    with open("results/mobilenetV3small_classification_report.txt", "w") as f:
        f.write(test_report)
        

if __name__ == "__main__":
    main()
