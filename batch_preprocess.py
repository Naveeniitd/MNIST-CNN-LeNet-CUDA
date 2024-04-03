import cv2
import numpy as np
import os

# Parameters
input_dir = 'test'
output_dir = 'batch_binary'
batch_size = 100  # Number of images per batch

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize
batch_images = []
file_names = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
total_batches = len(file_names) // batch_size + (1 if len(file_names) % batch_size > 0 else 0)

for batch_idx in range(total_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(file_names))
    batch_files = file_names[start_idx:end_idx]

    for file_name in batch_files:
        file_path = os.path.join(input_dir, file_name)
        img = cv2.imread(file_path, 0)
        if img.shape != [28, 28]:
            img = cv2.resize(img, (28, 28))
        img = img.reshape(28, 28, -1).astype(np.float32) / 255.0
        batch_images.append(img)

    # Convert list of arrays to a single 3D numpy array (batch_size, 28, 28)
    batch_images_np = np.array(batch_images)
    # Reset for next batch
    batch_images = []

    # Construct output file path for the batch
    output_file_path = os.path.join(output_dir, f'batch_{batch_idx}.bin')
    # Save the batch to a .bin file
    batch_images_np.tofile(output_file_path)

    print(f"Processed and saved batch: {output_file_path}")
