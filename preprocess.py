import cv2
import numpy as np

# Set NumPy print options
np.set_printoptions(linewidth=np.inf, formatter={'float': '{: 0.6f}'.format})

# Read the image in grayscale
img = cv2.imread('2.png', 0)

# Resize the image to 28x28 if it's not already that size
if img.shape != (28, 28):
    img = cv2.resize(img, (28, 28))

# Normalize and invert the image
img = 1.0 - img / 255.0

# Save the preprocessed image to a binary file
img.tofile('input.bin')
