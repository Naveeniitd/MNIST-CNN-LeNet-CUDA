import cv2
import numpy as np

# Load and preprocess the image
img_path = '2.png'
img = cv2.imread(img_path, 0)
if img.shape != (28, 28):
    img = cv2.resize(img, (28, 28))

# Revert the image and normalize it to the 0-1 range
img = 1.0 - img / 255.0

# Flatten the image to create a 1D array and ensure it's in float32 for CUDA
img_flattened = img.flatten().astype(np.float32)

# Save the flattened image to a binary file
binary_path = '2.bin'
img_flattened.tofile(binary_path)
