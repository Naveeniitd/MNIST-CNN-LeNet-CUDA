# Image Processing Library for MNIST Digit Recognition

This project implements an image processing library in C++ and CUDA aimed at recognizing hand-written digits from the MNIST dataset. It includes a series of subtasks from basic image processing functions to the implementation of a Convolutional Neural Network (CNN) using the LeNet-5 architecture, followed by optimization using CUDA streams.


# Data Preprocessing

Before running the main components of the project, it's essential to preprocess your dataset. This project includes two Python scripts (batch_preprocess.py && preprocess.py) for converting images from the MNIST dataset into a binary format suitable for further processing.

### Batch Processing

**Purpose**: Prepares images for batch processing by converting them into a normalized binary format and organizing them into batches.
- **Input**: Images from the `img` directory.
- **Output**: Binary files in `pre-proc-img/batch_binary` and labels in `pre-proc-img/labels_batch_binary`.

**How to Use**:
```bash
python batch_preprocess.py
```
### Individual Testing

**Purpose**: Converts individual test images into a binary format, making them compatible with the neural network model.

- **Input**: Images from the `test` directory.
- **Output**: Binary files in `binary_test` directory.

**How to Use**:
```bash
python preprocess.py
```