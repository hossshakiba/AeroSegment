import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from PIL import Image


def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def mIoU(output, mask, smooth=1e-10, n_classes=23):
    with torch.no_grad():
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        output = output.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for cls in range(0, n_classes): # loop per pixel class
            true_class = output == cls
            true_label = mask == cls

            if true_label.long().sum().item() == 0: # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])


def save_predictions_as_imgs(model, loader, folder, device):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            outputs = model(x)
            outputs = torch.argmax(outputs, dim=1)

        for i in range(len(x)):
            input_image = x[i].detach().cpu().permute(1, 2, 0).numpy()
            ground_truth_image = y[i].detach().cpu()
            output_image = outputs[i].detach().cpu()

            # Create the figure and subplots
            fig, axes = plt.subplots(1, 3, figsize=(8, 4), dpi=100)
            fig.subplots_adjust(hspace=0.1, wspace=0.1)
            
            # Plot input image
            axes[0].imshow(input_image)
            axes[0].set_title("Image")
            axes[0].axis('off')
            
            # Plot segmentation output
            axes[1].imshow(output_image, cmap='viridis')
            axes[1].set_title("Segmentation")
            axes[1].axis('off')
            
            # Plot ground truth
            axes[2].imshow(ground_truth_image, cmap='viridis')
            axes[2].set_title("Ground Truth")
            axes[2].axis('off')
            
            # Save the figure
            save_path = os.path.join(folder, f'figure_{idx}_{i}.png')
            plt.savefig(save_path)
            
            plt.close(fig)