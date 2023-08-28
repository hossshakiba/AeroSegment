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

from models.model import UNET
from dataset import DroneDataset, create_df
from utils import (
    pixel_accuracy,
    mIoU,
    load_checkpoint,
    save_predictions_as_imgs
)


def test(model, criterion, data_loader, device):
    since = time.time()
    model.to(device)

    test_loss = 0.0
    test_mIoU_score = 0
    test_accuracy = 0

    model.eval()
    for inputs, targets in tqdm(data_loader["test_loader"], desc=f'Testing...'):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        test_mIoU_score += mIoU(outputs, targets)
        test_accuracy += pixel_accuracy(outputs, targets)

    test_loss = test_loss / len(data_loader["test_loader"])
    test_mIoU_score = test_mIoU_score / len(data_loader["test_loader"])
    test_accuracy = test_accuracy / len(data_loader["test_loader"])

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test mIoU: {test_mIoU_score:.4f}")

    time_elapsed = time.time() - since
    print(f'Inference completed in {time_elapsed // 60:.2f}m {time_elapsed % 60:.2f}s.')


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
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    IMAGE_HEIGHT = 256  # 6000 originally
    IMAGE_WIDTH = 256  # 4000 originally

    # Configurations
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    NUM_WORKERS = 2
    IMG_DIR = "../archive/dataset/semantic_drone_dataset/original_images/"
    MASK_DIR = "../archive/dataset/semantic_drone_dataset/label_images_semantic/"

    # Create dataframe of the data
    df = create_df(IMG_DIR)

    # Split data train_val/test
    X_train_val, X_test = train_test_split(df['id'].values, test_size=0.1, random_state=seed_value)

    t_test = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH, interpolation=cv2.INTER_NEAREST),
    ])

    test_set = DroneDataset(IMG_DIR, MASK_DIR, X_test, t_test)

    test_loader =  DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True),

    data_loaders = {
        "test_loader": DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True),
    }

    model = UNET().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    test(model, criterion, data_loaders, DEVICE)

    save_predictions_as_imgs(model, data_loaders["test_loader"], "../output_images", DEVICE)

