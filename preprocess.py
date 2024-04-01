import cv2
import numpy as np

np.set_printoptions(linewidth=np.inf,formatter={'float': '{: 0.6f}'.format})

img = cv2.imread('000008-num5.png',0)
if img.shape != [28,28]:
    img2 = cv2.resize(img,(28,28))
    
img = img2.reshape(28,28,-1);
# Ensure img is of type float32
img = img.astype(np.float32)

#revert the image,and normalize it to 0-1 range
img =img/255.0

img.tofile("000008-num5.bin")

# with open('7.bin', 'rb') as file:
#     data = np.fromfile(file, dtype=np.float32)  # Ensure dtype matches what you wrote
#     print(data[:500]) 