import time
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import albumentations as A
import cv2

from models.model import AeroSegment
from dataset import DroneDataset, create_df
from utils import (
    pixel_accuracy,
    mIoU,
    save_checkpoint
)


def train(model, optimizer, criterion, n_epoch, data_loaders: dict, device):
    since = time.time()
    model.to(device)
    best_mIoU = 0.0

    for epoch in range(n_epoch):
        train_loss = 0.0
        train_mIoU_score = 0
        train_accuracy = 0

        model.train()
        for inputs, targets in tqdm(data_loaders['train_loader'], desc=f'Training... Epoch: {epoch + 1}/{n_epoch}'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            train_loss += loss.item()
            train_mIoU_score += mIoU(outputs, targets)
            train_accuracy += pixel_accuracy(outputs, targets)

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(data_loaders['train_loader'])
        train_mIoU_score = train_mIoU_score / len(data_loaders['train_loader'])
        train_accuracy = train_accuracy / len(data_loaders['train_loader'])
        
        with torch.no_grad():
            val_loss = 0.0
            val_mIoU_score = 0
            val_accuracy = 0

            model.eval()
            for inputs, targets in tqdm(data_loaders['val_loader'], desc=f'Validating... Epoch: {epoch + 1}/{n_epoch}'):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                val_mIoU_score += mIoU(outputs, targets)
                val_accuracy += pixel_accuracy(outputs, targets)

            val_loss = val_loss / len(data_loaders['val_loader'])
            val_mIoU_score = val_mIoU_score / len(data_loaders['train_loader'])
            val_accuracy = val_accuracy / len(data_loaders['train_loader'])

            if val_mIoU_score > best_mIoU:
                best_mIoU = val_mIoU_score
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": val_loss,
                    "accuracy": val_accuracy,
                    "mIoU": val_mIoU_score
                }
                save_checkpoint(checkpoint)

        print(f"Epoch [{epoch+1}/{n_epoch}]:")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train mIoU: {train_mIoU_score:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation mIoU: {val_mIoU_score:.4f}")
        print('-'*30)

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60:.2f}m {time_elapsed % 60:.2f}s.')


if __name__=="__main__":
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    IMAGE_HEIGHT = 256  # 6000 originally
    IMAGE_WIDTH = 256  # 4000 originally

    # Configurations
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    NUM_WORKERS = 2
    IMG_DIR = "../archive/dataset/semantic_drone_dataset/original_images/"
    MASK_DIR = "../archive/dataset/semantic_drone_dataset/label_images_semantic/"

    # Create dataframe of the data
    df = create_df(IMG_DIR)

    # Split data train/val
    X_train_val, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=seed_value)
    X_train, X_val = train_test_split(X_train_val, test_size=0.1, random_state=seed_value)

    t_train = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(),
        A.VerticalFlip(), 
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast((0,0.5),(0,0.5)),
        A.GaussNoise(),
    ])

    t_val = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(),
        A.VerticalFlip(), 
        A.GridDistortion(p=0.2),
    ])

    train_set = DroneDataset(IMG_DIR, MASK_DIR, X_train, t_train)
    val_set = DroneDataset(IMG_DIR, MASK_DIR, X_val, t_val)

    data_loaders = {
        "train_loader": DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        "val_loader": DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    }

    model = AeroSegment(in_channels=3, out_channels=23).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, optimizer, criterion, NUM_EPOCHS, data_loaders, DEVICE)

