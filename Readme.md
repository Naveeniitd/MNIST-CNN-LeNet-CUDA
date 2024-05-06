
# Project Title

## Directory Structure

```
.
├── img                      # Directory to read images from
├── Makefile                 # Generates binaries, stored at the same level as this file
├── output                   # Directory to save outputs in the format provided in the assignment
├── preprocessing.py         # Script for preprocessing data
├── pre-proc-img             # Directory to store preprocessed data
├── README.md                # This file
├── Report.pdf               # Project report
├── src                      # Source files
│   ├── assignment2_subtask1.cpp
│   ├── assignment2_subtask2.cu
│   ├── assignment2_subtask3.cu
│   └── assignment2_subtask4.cu
├── subtask1                 # Binary for subtask 1
├── subtask2                 # Binary for subtask 2
├── subtask3                 # Binary for subtask 3
├── subtask4                 # Binary for subtask 4
└── weights                  # Weights for neural network layers
    ├── conv1.txt
    ├── conv2.txt
    ├── fc1.txt
    └── fc2.txt

```

## Usage Instructions

### Subtask 1: Operations on Matrices
Execute various operations by specifying the task of choice:

- **Convolution**
  ```
  ./subtask1 1 <matrix_size> <kernel_size> <padding> <matrix_values...> <kernel_values...>
  ```
- **Non-linear Activations**
  ```
  ./subtask1 2 <activation_func> <matrix_size> <matrix_values...>
  ```
- **Subsampling**
  ```
  ./subtask1 3 <pooling_func> <pooling_size> <matrix_size> <matrix_values...>
  ```
- **Vector Conversion**
  ```
  ./subtask1 4 <conversion_func> <vector_values...>
  ```

### Subtask 2: Enhanced Operations with GPU Acceleration
Similar to subtask 1, with execution format:
```
./subtask2 [task of choice 1=convolution, 2=non-linear-activations, 3=subsampling, 4=converting a vector]
```

### Subtask 3: Run Preprocessed Files
Assumes files are preprocessed using `preprocessing.py`. Outputs top 5 probabilities in the output folder:
```
./subtask3
```

### Subtask 4: Stream-Based Processing
Toggle between with or without streams:
```
./subtask4 [1 - with streams, 0 - without streams]
```

## Build
Use the provided Makefile to compile the project:
```
make
```

## Output Format
Outputs are saved in the `output` directory in the same format as specified in the [sample output](https://www.cse.iitd.ac.in/~rijurekha/col380_2024/2_softmax.txt).

## Additional Information
Refer to `Report.pdf` for detailed information on the implementation and evaluation of the project.


