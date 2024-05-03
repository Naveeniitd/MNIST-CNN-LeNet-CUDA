#include <vector>
#include <iostream>
#include <numeric>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cfloat>
#include <sstream>
#include <queue>
#include <dirent.h>
using namespace std;
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct Weights {
    float* conv2;
    float* fc1;
    float* conv1;
    float* fc2; 
};
void saveArrayToFile(const float* array, int size, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i) {
        file << array[i];
        if (i < size - 1) {
            file << std::endl; // or use 'file << " ";' for space-separated values
        }
    }

    file.close();
}
void printArray(float* array, int n) {
    std::cout << "[";
    for (int i = 0; i < n; ++i) {
        std::cout << array[i];
        if (i < n - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}
//----------------------------------------------Convolution without padding-----------------------------------------------------------------------------//
__global__ void conv_cuda(const float* input, const float* weights, float* output, int in_c, int out_c, int isize, int ksize, int pad) {
    int res = isize - ksize + 1 + 2* pad;
    // Calculate output coordinates (x, y, o)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int o = blockIdx.z * blockDim.z + threadIdx.z;

    if (o < out_c && x < res && y < res) {
        float sum = 0.0f;
        // Convolution operation
        for (int c = 0; c < in_c; ++c) {
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    int iy = y + ky - pad;
                    int ix = x + kx - pad;
                    if (ix >= 0 && ix < isize && iy >= 0  && iy < isize) {
                        sum += input[c * isize * isize + iy * isize + ix] *
                               weights[o * (in_c * ksize * ksize) + c * (ksize * ksize) + ky * ksize + kx];
                    }
                }
            }
        }
        output[o * res * res + y * res + x] = sum + weights[out_c * in_c * ksize * ksize + o];
    }
}
//-------------------------------------Relu------------------------------------------------------------------------------//
__global__ void relu(float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        input[i] = max(0.0f, input[i]);
    }
}
//---------------------------------------Tanh----------------------------------------------------------------------------//
__global__ void tanh(float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i< n) {
        input[i] = tanhf(input[i]); 
    }
}
//---------------------------------------------------Max Pooling----------------------------------------------------------//
__global__ void maxpool_cuda(const float* input, float* output, int in_c, int isize, int ksize, int stride) {
    int res = (isize - ksize) / stride + 1;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < res && y < res && c < in_c) {
        const float* curr_input = input + c * isize * isize;
        float* curr_output = output + c * res * res;

        float maxVal = -FLT_MAX;
        for (int ky = 0; ky < ksize; ++ky) {
            for (int kx = 0; kx < ksize; ++kx) {
                int iy = y * stride + ky;
                int ix = x * stride + kx;
                if (ix < isize && iy < isize) {
                    int input_idx = iy * isize + ix;
                    maxVal = max(maxVal, curr_input[input_idx]);
                }
            }
        }
        curr_output[y * res + x] = maxVal;
    }
}
//-------------------------------------------------------Average Pooling---------------------------------------------------------------//
__global__ void avgpool_cuda(const float* input, float* output, int in_c, int isize, int ksize, int stride) {
    int res = (isize - ksize) / stride + 1;
    
    // Determine the output x, y coordinates and the c this thread is responsible for
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < res && y < res && c < in_c) {
        const float* curr_input = input + c * isize * isize;
        float* curr_output = output + c * res * res;

        float sum = 0.0f;
        int count = 0; // Counter for valid pooling area elements

        // Pooling operation
        for (int ky = 0; ky < ksize; ++ky) {
            for (int kx = 0; kx < ksize; ++kx) {
                int iy = y * stride + ky; // Calculate the input y-coordinate for this kernel element
                int ix = x * stride + kx; // Calculate the input x-coordinate for this kernel element

                // Check if the coordinates are within the bounds of the input size
                if (ix < isize && iy < isize) {
                    int input_idx = iy * isize + ix;
                    sum += curr_input[input_idx];
                    count++;
                }
            }
        }

        // Calculate the average and write it to the output
        if (count > 0) { 
            curr_output[y * res + x] = sum / count;
        } else {
            curr_output[y * res + x] = 0.0f;
        }
    }
}

//------------------------------------------------------Fully connected Convolution-------------------------------------------------------------//
__global__ void fconv_cudaR(const float* input, const float* weights, float* output, int in_c, int out_c, int isize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int inputSize = isize * isize * in_c; // Total size of the input for one filter

    if (index < out_c) { // Check if thread index is within the range of output cs
        const float* curr_weights = weights + index * inputSize; // Adjust index for multi-c
        float bias = weights[out_c * inputSize + index]; // Bias index adjusted for flattened input
        float sum = 0.0f;

        for (int j = 0; j < inputSize; ++j) {
            sum += input[j] * curr_weights[j];
        }

        // Apply bias and ReLU activation
        output[index] = max(0.0f, sum + bias); // Using max for ReLU
    }
}
//---------------------------------------------------Fully connected Convolution without relu-----------------------------------------------------------//
__global__ void fconv_cuda(const float* input, const float* weights, float* output, int in_c, int out_c, int isize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int inputSize = isize * isize * in_c; // Total size of the input for one filter

    if (index < out_c) {
        const float* curr_weights = weights + index * inputSize; // Adjust index for multi-channels
        float bias = weights[out_c * inputSize + index]; // Bias index adjusted for flattened input
        float sum = 0.0f;

        for (int j = 0; j < inputSize; ++j) {
            sum += input[j] * curr_weights[j];
        }

        // Apply bias and ReLU activation
        output[index] = sum + bias; // Using max for ReLU
    }
}
//-----------------------------------------------------------Sigmoid-------------------------------------------------------------------------------------//
__global__ void sigmoid_cuda(float* arr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        arr[i] = 1.0f / (1.0f + exp(-arr[i]));
    }
}
//----------------------------------------------------------------softmax------------------------------------------------------------------------------//
void softmax(float* arr, int size) {
    float sum = 0.0;

    // Calculate the exponentials of each element and their sum
    for(int i = 0; i < size; i++) {
        arr[i] = exp(arr[i]);
        sum += arr[i];
    }

    // Normalize the array to get the softmax probabilities
    for(int i = 0; i < size; i++) {
        arr[i] /= sum;
    }
}
//--------------------------Function-to-read-trained-weights---------------------------//
float* fileRead(const string& path, int size) {
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << path << endl;
        return nullptr;
    }
    float* weights = new float[size];
    for (int i = 0; i < size; i++) {
        if (!(file >> weights[i])) {
            cerr << "Failed to read weight at position " << i << " from file: " << path << endl;
            delete[] weights; 
            return nullptr;
        }
    }
    file.close();
    return weights;
}
int main(int argc, char* argv[]){
    if (argc < 4) {
        // std::cerr << "Usage: " << argv[0] << " [path_to_bin_file] [Image Size] [in_channel] [Function] [different parameter for different function]" << std::endl;
        return 1;
    }
    int function = stoi(argv[1]);
    if(function == 1){
        int isize = stoi(argv[2]);
        int ksize = stoi(argv[3]);
        int pad = stoi(argv[4]);
        float* input = new float[isize * isize];
        float* kernel = new float[ksize * ksize];
        for(int i = 5 ; i <  (isize*isize)+5; i++){
            input[i-5] = stof(argv[i]);
        }
        for(int k = (isize*isize)+5 ; k <  (ksize*ksize) + ((isize*isize)+5); k++){
            kernel[k-((isize*isize)+5)] = stof(argv[k]);
        }
        int res = isize - ksize + 2 * pad + 1;
        
        float *c_kernel, *c_input, *c_output;
        CHECK_CUDA_ERROR(cudaMalloc(&c_input, isize*isize * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(c_input, input, isize*isize * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMalloc(&c_kernel, ksize*ksize * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMemcpy(c_kernel, kernel, ksize*ksize * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMalloc(&c_output, res*res * sizeof(float)));
        dim3 c1_block(16, 16, 1); 
        dim3 c1_grid((res + c1_block.x - 1) / c1_block.x,(res + c1_block.y - 1) / c1_block.y,1);
        conv_cuda<<<c1_grid, c1_block>>>(c_input, c_kernel, c_output, 1, 1, isize, ksize, pad);
        cudaDeviceSynchronize(); // Wait for the kernel to complete
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "Error during conv_cuda_1 execution: " << cudaGetErrorString(error) << std::endl;
        }
        float* output  = new float[res*res];
        CHECK_CUDA_ERROR(cudaMemcpy(output, c_output, res*res * sizeof(float), cudaMemcpyDeviceToHost));
        printArray(output, res*res);
        CHECK_CUDA_ERROR(cudaFree(c_output));
        CHECK_CUDA_ERROR(cudaFree(c_kernel));
        CHECK_CUDA_ERROR(cudaFree(c_input));
        delete[] input;
        delete[] output;
        delete[] kernel;
        
    }
    else if (function == 2 ){ 
        int act = stoi(argv[2]);
        if(act == 0){
            int isize = stoi(argv[3]);
            int ksize = stoi(argv[4]);
            float* input = new float[isize * ksize];
            for(int k = 5; k <  (isize * ksize) + 5; k++){
                input[k-5] = stof(argv[k]);
            }
            float *c_input;
            CHECK_CUDA_ERROR(cudaMalloc(&c_input, isize*ksize * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemcpy(c_input, input, isize*ksize * sizeof(float), cudaMemcpyHostToDevice));
            int r_block = 256;
            int r_grid = (isize + r_block - 1) / r_block;
            dim3 threads(r_block);
            dim3 blocks(r_grid);
            relu<<<blocks, threads>>>(c_input, isize*ksize);
            cudaDeviceSynchronize(); // Wait for the kernel to complete
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "Error during conv_cuda_1 execution: " << cudaGetErrorString(error) << std::endl;
            }
            float* output  = new float[isize*ksize];
            CHECK_CUDA_ERROR(cudaMemcpy(output, c_input, isize*ksize * sizeof(float), cudaMemcpyDeviceToHost));
            printArray(output, isize * ksize);
            delete[] input;
            delete[] output;
            CHECK_CUDA_ERROR(cudaFree(c_input));


        }else if(act ==1){
            int isize = stoi(argv[3]);
            int ksize = stoi(argv[4]);
            float* input = new float[isize * ksize];
            for(int k = 5; k <  (isize * ksize) + 5; k++){
                input[k-5] = stof(argv[k]);
            }
            float *c_input;
            CHECK_CUDA_ERROR(cudaMalloc(&c_input, isize*ksize * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemcpy(c_input, input, isize*ksize * sizeof(float), cudaMemcpyHostToDevice));
            int r_block = 256;
            int r_grid = (isize + r_block - 1) / r_block;
            dim3 threads(r_block);
            dim3 blocks(r_grid);
            tanh<<<blocks, threads>>>(c_input, isize*ksize);
            cudaDeviceSynchronize(); // Wait for the kernel to complete
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "Error during conv_cuda_1 execution: " << cudaGetErrorString(error) << std::endl;
            }
            float* output  = new float[isize*ksize];
            CHECK_CUDA_ERROR(cudaMemcpy(output, c_input, isize*ksize * sizeof(float), cudaMemcpyDeviceToHost));
            printArray(output, isize * ksize);
            delete[] input;
            delete[] output;
            CHECK_CUDA_ERROR(cudaFree(c_input));
            
        }
    }
    else if (function == 3){
        int pool = stoi(argv[2]);
        if(pool ==0){
            int psize = stoi(argv[3]);
            int isize = stoi(argv[4]);
            int res = (isize - psize) + 1;
            float* input = new float[isize * isize];
            float* output = new float[res*res];
            for(int k = 5; k <  (isize * isize ) + 5; k++){
                input[k-5] = stof(argv[k]);
            }
            float *c_input, *c_output;
            CHECK_CUDA_ERROR(cudaMalloc(&c_input, isize*isize * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&c_output, res*res * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemcpy(c_input, input, isize*isize * sizeof(float), cudaMemcpyHostToDevice));
            dim3 m_block(16, 16, 1);
            dim3 m_grid((res + m_block.x - 1) / m_block.x,(res + m_block.y - 1) / m_block.y,1);
            maxpool_cuda<<<m_grid, m_block>>>(c_input, c_output, 1, isize, psize, 1);
            cudaDeviceSynchronize();
            cudaError_t error1 = cudaGetLastError();
            if (error1 != cudaSuccess) {
                std::cerr << "Error during Max_pool_1 execution: " << cudaGetErrorString(error1) << std::endl;
            }
            CHECK_CUDA_ERROR(cudaMemcpy(output, c_output, res * res * sizeof(float), cudaMemcpyDeviceToHost));

            printArray(output, res*res);
            delete[] input;
            delete[] output;
            CHECK_CUDA_ERROR(cudaFree(c_output));
            CHECK_CUDA_ERROR(cudaFree(c_input));
        }else if(pool ==1){
            int psize = stoi(argv[3]);
            int isize = stoi(argv[4]);
            int res = (isize - psize) + 1;
            float* input = new float[isize * isize];
            float* output = new float[res*res];
            for(int k = 5; k <  (isize * isize ) + 5; k++){
                input[k-5] = stof(argv[k]);
            }
            float *c_input, *c_output;
            CHECK_CUDA_ERROR(cudaMalloc(&c_input, isize*isize * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMalloc(&c_output, res*res * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemcpy(c_input, input, isize*isize * sizeof(float), cudaMemcpyHostToDevice));
            dim3 m_block(16, 16, 1);
            dim3 m_grid((res + m_block.x - 1) / m_block.x,(res + m_block.y - 1) / m_block.y,1);
            avgpool_cuda<<<m_grid, m_block>>>(c_input, c_output, 1, isize, psize, 1);
            cudaDeviceSynchronize();
            cudaError_t error1 = cudaGetLastError();
            if (error1 != cudaSuccess) {
                std::cerr << "Error during Max_pool_1 execution: " << cudaGetErrorString(error1) << std::endl;
            }
            CHECK_CUDA_ERROR(cudaMemcpy(output, c_output, res * res * sizeof(float), cudaMemcpyDeviceToHost));

            printArray(output, res*res);
            delete[] input;
            delete[] output;
            CHECK_CUDA_ERROR(cudaFree(c_output));
            CHECK_CUDA_ERROR(cudaFree(c_input));
        }
    }
    else if (function == 4 ){
        int func = stoi(argv[2]);
        if(func ==0){
            int isize = argc - 3;
            //cout << isize << endl;
            float* input = new float[isize];
            for(int k = 3; k <  isize + 3; k++){
                input[k-3] = stof(argv[k]);
            }
            float *c_input;
            CHECK_CUDA_ERROR(cudaMalloc(&c_input, isize * sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemcpy(c_input, input, isize * sizeof(float), cudaMemcpyHostToDevice));
            int r_block = 256;
            int r_grid = (isize + r_block - 1) / r_block;
            dim3 threads(r_block);
            dim3 blocks(r_grid);
            sigmoid_cuda<<<blocks, threads>>>(c_input, isize);
            cudaDeviceSynchronize(); // Wait for the kernel to complete
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::cerr << "Error during conv_cuda_1 execution: " << cudaGetErrorString(error) << std::endl;
            }
            float* output  = new float[isize];
            CHECK_CUDA_ERROR(cudaMemcpy(output, c_input, isize * sizeof(float), cudaMemcpyDeviceToHost));
            printArray(output, isize);
            delete[] input;
            delete[] output;
            CHECK_CUDA_ERROR(cudaFree(c_input));
          
            
        }else if(func ==1){
            int isize = argc - 3;
            //cout << isize << endl;
            float* input = new float[isize];
            for(int k = 3; k <  isize + 3; k++){
                input[k-3] = stof(argv[k]);
            }
            softmax(input, isize);
            printArray(input, isize);
            delete[] input;
        }
    }
    return 0;

}
