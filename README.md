# Semantic Segmentation Project

This repository contains the implementation of a semantic segmentation model using the UNET architecture with the Channel-wise Attention Module (CBAM) for improved feature learning. The model aims to perform semantic understanding of urban scenes from aerial images, specifically for enhancing the safety of autonomous drone flight and landing procedures.

## Dataset Overview

The **Semantic Drone Dataset** is employed for training and evaluation. The dataset focuses on providing semantic understanding of urban scenes, captured from a bird's eye view using a high-resolution camera. The key details of the dataset are as follows:

- The dataset comprises imagery from more than 20 houses captured at altitudes ranging from 5 to 30 meters above the ground.
- Images are acquired with a high-resolution camera at a size of 6000x4000 pixels (24 megapixels).
- The training set includes 400 publicly available images, while the test set consists of 200 private images.

## Model Architecture

The implemented model is based on the UNET architecture, which has shown significant success in various semantic segmentation tasks. Additionally, the model incorporates the Channel-wise Attention Module (CBAM) to enhance feature representation. Here's a brief overview of the model's architecture:

### DoubleConv

The `DoubleConv` module consists of two consecutive convolutional layers, each followed by batch normalization and ReLU activation. CBAM attention is applied after the second convolutional layer.

### UNET

The `UNET` model is composed of an encoder and a decoder. The encoder consists of several `DoubleConv` blocks, which progressively downsample the input image. The decoder contains transposed convolutional layers that upsample the encoded features back to the original image size. Skip connections are established between the encoder and decoder to facilitate fine-grained feature propagation.

The model's architecture can be summarized as follows:
- Encoder: A stack of `DoubleConv` blocks with increasing feature channels.
- Bottleneck: Another `DoubleConv` block applied at the bottleneck of the model.
- Decoder: A series of transposed convolutional layers combined with skip connections from the encoder.
- Final Layer: A 1x1 convolutional layer producing the final segmented output.

## Usage

To use this repository and train the semantic segmentation model on your own data, follow these steps:

1. Clone this repository to your local machine.
2. Prepare your dataset by organizing it into appropriate training and testing subsets.
3. Replace the dummy data placeholders in the code with your actual dataset.
4. Install the required dependencies by running `pip install -r requirements.txt`.
5. Train the model using the provided training script: `python train.py`.
6. Evaluate the trained model using the evaluation script: `python evaluate.py`.

## Credits

This project is built upon the foundational concepts of UNET architecture and incorporates the Channel-wise Attention Module (CBAM). The UNET architecture was introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation."

The CBAM attention module was proposed in the paper "CBAM: Convolutional Block Attention Module" by Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon.

## License

This project is licensed under the [MIT License](LICENSE).
