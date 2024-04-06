import cv2
import numpy as np
import os

# Set input and output directory paths
input_dir = 'test'
output_dir = 'binary_test'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.set_printoptions(linewidth=np.inf, formatter={'float': '{: 0.6f}'.format})

# List all files in the input directory
for file_name in os.listdir(input_dir):
    # Check if the file is an image (e.g., '.png', '.jpg')
    if file_name.lower().endswith(('.png')):
        # Construct the full file path
        file_path = os.path.join(input_dir, file_name)
        
        # Read the image in grayscale
        img = cv2.imread(file_path, 0)
        
        # Resize the image if it's not already 28x28
        if img.shape != [28, 28]:
            img = cv2.resize(img, (28, 28))
        
        # Reshape the image
        img = img.reshape(28, 28, -1)
        # Convert image to float32
        img = img.astype(np.float32)
        # Normalize the image to 0-1 range
        img = img / 255.0
        
        # Construct the output file path
        output_file_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.bin')
        
        # Save the processed image to a .bin file
        img.tofile(output_file_path)

        print(f"Processed and saved: {output_file_path}")
