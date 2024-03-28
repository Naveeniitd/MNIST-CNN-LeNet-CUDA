import cv2
import numpy as np
import glob
import os

def preprocess_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, 0)

    # Resize the image to 28x28 if it's not already that size
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))

    # Normalize and invert the image
    img2 = 1.0 - img / 255.0

    return img2

def save(img, output_path):
    # Save the preprocessed image to a binary file
    img.tofile(output_path)


pattern = "test/*.png"
out_dir = "Binary_img"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


for image_path in glob.glob(pattern):
    img2 = preprocess_image(image_path)
    base_name = image_path.split('/')[-1].split('.')[0]
    output_path = os.path.join(out_dir, f"{base_name}.bin")
    save(img2, output_path)
