import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
import albumentations as A

import cv2
from tqdm import tqdm

from model import UNET
from dataset import DroneDataset, create_df


def train(model, optimizer, criterion, n_epoch,
          data_loaders: dict, device, lr_scheduler=None
          ):
    since = time.time()
    train_losses = np.zeros(n_epoch)
    val_losses = np.zeros(n_epoch)

    model.to(device)

    for epoch in range(n_epoch):
        train_loss = 0.0

        model.train()
        for inputs, targets in tqdm(data_loaders['train_loader'], desc=f'Training... Epoch: {epoch + 1}/{n_epoch}'):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(data_loaders['train_loader'])
        
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            
            for inputs, targets in tqdm(data_loaders['val_loader'], desc=f'Validating... Epoch: {epoch + 1}/{n_epoch}'):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # # Save output images every 20 epoch
                # if (epoch + 1) % 20 == 0:
                #     save_output_images(outputs, SAVE_DIR_ROOT, epoch, MODEL_NAME, device)

            val_loss = val_loss / len(data_loaders['val_loader'])

            # if val_psnr > best_psnr:
                # save_model(model, optimizer, val_loss, val_psnr,
                #            val_ssim, epoch, SAVE_DIR_ROOT, MODEL_NAME, device)
                # save_output_images(outputs, SAVE_DIR_ROOT, epoch, MODEL_NAME, device, True)

        # save epoch losses
        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss

        print(f"Epoch [{epoch+1}/{n_epoch}]:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print('-'*20)

    time_elapsed = time.time() - since
    print(f'Training completed in {time_elapsed // 60}m {time_elapsed % 60}s.')


if __name__=="__main__":
    # Hyperparameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    IMAGE_HEIGHT = 100  # 6000 originally
    IMAGE_WIDTH = 100  # 4000 originally

    # Configurations
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    NUM_WORKERS = 2
    IMG_DIR = "../archive/dataset/semantic_drone_dataset/original_images/"
    MASK_DIR = "../archive/dataset/semantic_drone_dataset/label_images_semantic/"

    # Create dataframe of the data
    df = create_df(IMG_DIR)

    # Split data train/val/test
    X_train_val, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=42)
    X_train, X_val = train_test_split(X_train_val, test_size=0.1, random_state=42)

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
        A.GridDistortion(p=0.2),
    ])

    train_set = DroneDataset(IMG_DIR, MASK_DIR, X_train, t_train)
    val_set = DroneDataset(IMG_DIR, MASK_DIR, X_val, t_val)

    data_loaders = {
        "train_loader": DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        "val_loader": DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    }


    model = UNET().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, optimizer, criterion, NUM_EPOCHS, data_loaders, DEVICE)

