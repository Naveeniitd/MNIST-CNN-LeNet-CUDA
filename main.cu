#include <vector>
#include <iostream>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
using namespace std;

// void printMatrix(const vector<vector<float> >& matrix) {
//     for (const auto& row : matrix) {
//         for (const auto& elem : row) {
//             cout << elem << " ";
//         }
//         cout << endl;
//     }
// }
__global__ void conv(const float* input, const float* kernel, float* output, int isize, int ksize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int res = isize - ksize + 1;

    if (i < res && j < res) {
        float sum = 0.0f;
        for (int m = 0; m < ksize; ++m) {
            for (int n = 0; n < ksize; ++n) {
                int X = i + m;
                int Y = j + n;
                sum += input[Y * isize + X] * kernel[m * ksize + n];
            }
        }
        output[j * res + i] = sum;
    }
}


// vector<vector<float> > convpad(vector<vector<float> > input, vector<vector<float> > kernel){
//     int isize = input.size();
//     int ksize = kernel.size();
//     int pad = (ksize-1)/2;
//     int padinput = isize + pad*2;
//     int res = isize;

//     vector<vector<float> > padMatrix(padinput, vector<float>(padinput, 0));

//     vector<vector<float> > outputMatrix(res, vector<float>(res, 0));

//     for (int i = 0; i < isize; ++i) {
//         for (int j = 0; j < isize; ++j) {
//             padMatrix[i + pad][j + pad] = input[i][j];
//         }
//     }
//     printMatrix(padMatrix);

//     for (int i = 0; i < res; ++i) {
//         for (int j = 0; j < res; ++j) {
//             float sum = 0;
//             for (int m = 0; m < ksize; ++m) {
//                 for (int n = 0; n < ksize; ++n) {
//                     sum += padMatrix[i + m][j + n] * kernel[m][n];
//                 }
//             }
//             outputMatrix[i][j] = sum;
//         }
//     }

//     return outputMatrix;
// }

__global__ void relu(float* input, float* output, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        output[idx] = max(0.0f, input[idx]);
    }
}

__global__ void tanh(float* input, int dataSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dataSize) {
        input[idx] = tanhf(input[idx]); // Note: Use tanhf for float
    }
}


// __global__ void maxPool(const float* input, float* output, int isize, int poolSize, int outputSize) {
//     int ox = blockIdx.x * blockDim.x + threadIdx.x; // Output x-coordinate
//     int oy = blockIdx.y * blockDim.y + threadIdx.y; // Output y-coordinate

//     if (ox < outputSize && oy < outputSize) {
//         float maxVal = -FLT_MAX;
//         for (int i = 0; i < poolSize; ++i) {
//             for (int j = 0; j < poolSize; ++j) {
//                 int ix = ox * poolSize + i;
//                 int iy = oy * poolSize + j;
//                 maxVal = max(maxVal, input[iy * isize + ix]);
//             }
//         }
//         output[oy * outputSize + ox] = maxVal;
//     }
// }


float sigmoid(float x){
    return 1.0f/(1.0f+exp(-x));
}

vector<float> sigfunc(const vector<float>& input){
    vector<float> output(input.size());
    transform(input.begin(), input.end(), output.begin(), sigmoid);
    return output;
}


// vector<float> softmax(const vector<float>& input) {
//     vector<float> outputMatrix(input.size());
//     float p = *max_element(input.begin(), input.end());
//     float sum = 0.0f;
//     for (int i = 0; i < input.size(); ++i) {
//         outputMatrix[i] = exp(input[i] - p); 
//         sum += outputMatrix[i];
//     }
//     for (float& value : outputMatrix) {
//         value /= sum;
//     }

//     return outputMatrix;
// }
vector<vector<float> > fileread(ifstream& file) {
    int rows, cols;
    file >> rows >> cols; 
    vector<vector<float> > matrix(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix[i][j];
        }
    }
    return matrix;
}

int main() {
    ifstream file("matrix.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return -1;
    }
    
    vector<vector<float> > input = fileread(file);
    vector<vector<float> > kernel = fileread(file);
    int isize = input.size();
    int ksize = kernel.size();
    
    vector<vector<float> > outputMatrix(res, vector<float>(res, 0));
    int res = isize - ksize +1;
    size_t inputSize = isize * isize * sizeof(float);
    size_t kernelsize = ksize*ksize*sizeof(float);
    size_t outputsize = res*res*sizeof(float);

    
    float *c_input, *c_kernel, *c_output;
    cudaMalloc(&c_input, inputSize);
    cudaMalloc(&c_kernel, kernelsize);
    cudaMalloc(&c_output, outputsize);


    cudaMemcpy(c_input, input, inputsize, cudaMemcpyHostToDevice);
    cudaMemcpy(c_kernel, kernel, kernelsize, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((res + threads.x - 1) / threads.x, 
                   (res + threads.y - 1) / threads.y);

    conv<<<blocks, threads>>>(c_input, c_kernel, c_output, isize, ksize);

    cudaMemcpy(outputMatrix, c_output, outputsize, cudaMemcpyDeviceToHost);
    cudaFree(c_input);
    cudaFree(c_kernel);
    cudaFree(c_output);

    printMatrix(outputMatrix);

    return 0;
}

