import cv2
import numpy as np
import os
import struct

input_dir = 'img'
output_dir = 'pre-proc-img/'
batch_size = 100

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize
file_names = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
total_batches = len(file_names) // batch_size + (1 if len(file_names) % batch_size > 0 else 0)

for batch_idx in range(total_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(file_names))
    batch_files = file_names[start_idx:end_idx]

    
    output_file_path = os.path.join(output_dir, f'batch_{batch_idx}.bin')

    with open(output_file_path, 'wb') as f:
        for file_name in batch_files:
            file_path = os.path.join(input_dir, file_name)
            img = cv2.imread(file_path, 0)
            if img is None:
                print(f"Warning: Could not load image {file_name}. Skipping.")
                continue
            
            if img.shape != (28, 28):
                img = cv2.resize(img, (28, 28))
            img_flat = img.ravel().astype(np.float32) / 255.0
            
            # Write the filename length and filename
            encoded_filename = file_name.encode('utf-8')
            f.write(struct.pack('I', len(encoded_filename)))
            f.write(encoded_filename)
            
            # Write the image data
            data_format = f'{img_flat.size}f'
            f.write(struct.pack(data_format, *img_flat))

    
