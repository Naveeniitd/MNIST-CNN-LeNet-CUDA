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

# Subtask 1: Running C++ Functions

This C++ library provides a set of functions for preprocessing and analyzing MNIST digit images. It includes convolution operations (with and without padding), non-linear activation functions (ReLU, tanh), subsampling (max pooling, average pooling), and transformation of float vectors to probability vectors using softmax and sigmoid functions. The library is designed to support flexible command-line arguments for easy integration into broader machine learning or image processing pipelines.

## Getting Started

### Prerequisites

Ensure you have access to an HPC cluster with CUDA support. Load any necessary modules, such as CUDA and a compatible compiler, if required by your HPC environment.

### Compilation

To compile the code, navigate to the project directory and use the following command:

```bash
module load compiler/cuda/10.2/compilervars
nvcc -std=c++11 assignment2_subtask1.cu -o s1
nvcc -std=c++11 assignment2_subtask2.cu -o s2
```

This command compiles the source code into an executable named `s1`/`s2`.

## Usage

The `s1`/`s2` executable supports various command-line arguments to specify the operation to perform and its parameters. Here's how to use it:

```bash
./s1 [path_to_bin_file] [Image Size] [in_channel] [Function] [additional parameters]
```

- `[path_to_bin_file]`: Path to the binary file containing the image data.
- `[Image Size]`: Size of the image (e.g., 28 for MNIST).
- `[in_channel]`: Number of input channels (e.g., 1 for grayscale images).
- `[Function]`: The function to apply. Options include `MaxPooling`, `AvgPooling`, `Softmax`, `Sigmoid`, `relu`, and `tanh`.
- `[additional parameters]`: Depends on the function chosen. For pooling operations, specify `[ksize] [stride]`.

### Examples

#### Max Pooling

```bash
./s1 ./data/2.bin 28 1 MaxPooling 2 2
```

This command applies max pooling to the image in `2.bin` with a kernel size of 2 and a stride of 2.

## Functions Implemented

- **Convolution**: Applies a convolution operation between an input matrix and a kernel.
- **ReLU and Tanh**: Applies the ReLU or tanh activation function to an input matrix.
- **Max Pooling and Average Pooling**: Subsamples an input matrix using the max or average pooling method.
- **Softmax and Sigmoid**: Converts a vector of floats to a vector of probabilities using the softmax or sigmoid function.



