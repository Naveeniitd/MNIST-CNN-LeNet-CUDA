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

# Subtask 1/2: Running C++ Functions/ Cuda Kernels

This comprehensive C++ and CUDA library is designed for efficient image processing and recognition of handwritten digits from the MNIST dataset. By leveraging the parallel computing power of NVIDIA GPUs, it offers high-performance implementations of critical operations involved in deep learning models, particularly those used in Convolutional Neural Networks (CNNs).

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
- `[Function]`: The function to apply. Options include `max`, `avg`, `soft`, `sig`, `relu`, and `tanh`.
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


## Verifying CUDA and C++ Outputs

To ensure the CUDA kernels' outputs match those from C++ functions, use the `check_matrix.py` Python script. This script compares output files line-by-line to check for consistency.

### Steps:

1. **Generate Outputs**: Run both your CUDA and C++ implementations to generate output files.

2. **Run Comparison Script**: 

    ```bash
    python chech_matrix.py
    ```

3. **Review Results**: The script will indicate whether the files are "the same" or "different."



# Subtask 3/4: LeNet-5 CNN Implementation/ CUDA Stream Optimization

This section of our library harnesses the power of NVIDIA CUDA for rapid image processing and digit recognition on the MNIST dataset, employing the LeNet-5 architecture. It features convolutional layers, activation functions (ReLU, Tanh), pooling layers (Max and Average), and fully connected layers, culminating in a softmax probability distribution for accurate digit classification.

The implementation optimizes the MNIST digit recognition process using CUDA streams, enhancing throughput and efficiency significantly. CUDA streams facilitate the concurrent execution of kernel launches and memory transfers, enabling our library to process images at high speed without compromising accuracy or performance.

By adopting CUDA streams, we streamline the workflow, allowing for simultaneous operations that leverage GPU resources more effectively. This optimization ensures faster processing times for the entire MNIST dataset, showcasing the potential of parallel computing in deep learning tasks.

## Running the Digit Recognition Process

Before executing the recognition process, ensure your environment is set up with the necessary CUDA toolkit and that your NVIDIA GPU driver is correctly installed. Additionally, ensure the MNIST dataset is preprocessed into binary format as expected by the program.

1. **Compilation**: To compile the program, navigate to the source directory and use the `nvcc` compiler:
    ```bash
    nvcc -std=c++11 -o s3 assignment2_subtask3.cu
    nvcc -std=c++11 -o s4 assignment2_subtask4.cu
    ```

2. **Execution**: Run the compiled executable with the following command:
    ```bash
    ./s3
    ./s4
    ```
    The program automatically processes the MNIST images located in the predefined directory and outputs the recognition results.

### Directory Structure

- **Weights Directory**: Contains pre-trained weights for the LeNet-5 model layers. Ensure the paths in the source code match the location of these files on your system.
- **Output Directory**: The program saves the top 5 softmax probabilities for each processed MNIST image here, along with the predicted class labels.

### Output Interpretation

For each image processed, the program generates a file containing the top 5 predictions in descending order of probability. Each line in the output file corresponds to a class (digit) and its associated probability, formatted as follows:
```
Class 2 Probability: 0.999
Class 1 Probability: 0.0005
...
```
### Performance Metrics

The program calculates and displays the overall accuracy of the recognition process, comparing the top prediction for each image against the true labels. Additionally, it reports the total processing time, allowing for performance assessments. Labels are given on the name of the image such that : `000017-num7.png` with `7` as Label.

### Accuracy and Time Measurement

Upon completion, the program prints the accuracy percentage and total elapsed time (in milliseconds) to the console:
```
Accuracy of result is 99.15%
Time taken: 12345 ms
```
