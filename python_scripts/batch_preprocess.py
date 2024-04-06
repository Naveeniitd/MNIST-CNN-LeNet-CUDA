import cv2
import numpy as np
import os

# Parameters
input_dir = 'img'
output_dir = 'pre-proc-img/batch_binary'
output_dir_label = 'pre-proc-img/labels_batch_binary'
batch_size = 100  # Number of images per batch

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir_label):
    os.makedirs(output_dir_label)

# Initialize
batch_images = []
batch_labels = []  # To store labels for each batch
file_names = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
total_batches = len(file_names) // batch_size + (1 if len(file_names) % batch_size > 0 else 0)

for batch_idx in range(total_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(file_names))
    batch_files = file_names[start_idx:end_idx]

    for file_name in batch_files:
        # Extract label from filename
        label = int(file_name.split('-num')[-1].split('.')[0])  # Assumes label is a single digit
        batch_labels.append(label)
        
        # Process image
        file_path = os.path.join(input_dir, file_name)
        img = cv2.imread(file_path, 0)
        if img.shape != [28, 28]:
            img = cv2.resize(img, (28, 28))
        img = img.reshape(28, 28, -1).astype(np.float32) / 255.0
        batch_images.append(img)

    # Convert list of arrays to a single 3D numpy array (batch_size, 28, 28) for images
    batch_images_np = np.array(batch_images)
    # Convert list of labels to a numpy array
    batch_labels_np = np.array(batch_labels)

    # Reset for next batch
    batch_images = []
    batch_labels = []

    # Construct output file paths for the batch
    output_images_file_path = os.path.join(output_dir, f'batch_{batch_idx}.bin')
    output_labels_file_path = os.path.join(output_dir_label, f'batch_{batch_idx}_labels.txt')

    # Save the batch images to a .bin file
    batch_images_np.tofile(output_images_file_path)
    # Save the labels to a .txt file
    np.savetxt(output_labels_file_path, batch_labels_np, fmt='%d')

    print(f"Processed and saved batch: {output_images_file_path} with labels: {output_labels_file_path}")
