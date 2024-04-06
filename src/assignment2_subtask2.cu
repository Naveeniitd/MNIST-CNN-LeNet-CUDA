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
__global__ void conv_cuda(const float* input, const float* weights, float* output, int in_c, int out_c, int isize, int ksize) {
    int res = isize - ksize + 1;
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
                    int iy = y + ky;
                    int ix = x + kx;
                    if (ix < isize && iy < isize) {
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
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [path_to_bin_file] [Image Size] [in_channel] [Function] [different parameter for different function]" << std::endl;
        return 1;
    }
    Weights weights;
    weights.conv1 = fileRead("/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/weights/conv1.txt", 520);
    weights.conv2 = fileRead("/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/weights/conv2.txt", 25050);
    weights.fc1 = fileRead("/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/weights/fc1.txt", 400500);
    weights.fc2 = fileRead("/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/weights/fc2.txt", 5010);
    float *d_conv1, *d_fc2, *d_conv2, *d_fc1; //variable for Device holding pointer to the data of conv1.txt, conv2.txt fc1.txt and fc2.txt respectively
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv1, 520 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv1, weights.conv1, 520 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv2, 25050 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv2, weights.conv2, 25050 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc1, 400500 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc1, weights.fc1, 400500 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc2, 5010 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc2, weights.fc2, 5010 * sizeof(float), cudaMemcpyHostToDevice));
    //printArray(weights.conv1, 520);
    int isize = stoi(argv[2]);
    int in_channel = stoi(argv[3]);
    
    float* input = new float[isize * isize*in_channel];
    std::ifstream file(argv[1], std::ios::binary);
    
    if (!file) {
        std::cerr << "Cannot open file!" << std::endl;
        return 1;
    }
   
    // Read the entire image data into the array
    file.read(reinterpret_cast<char*>(input), isize*isize*in_channel*sizeof(float));
    if (!file) {
        std::cerr << "Error reading file or file too short!" << std::endl;
        return 2;
    }

    file.close();


    float* k_input;
    CHECK_CUDA_ERROR(cudaMalloc(&k_input, isize*isize*in_channel* sizeof(float))); 
    CHECK_CUDA_ERROR(cudaMemcpy(k_input, input,isize*isize*in_channel*sizeof(float), cudaMemcpyHostToDevice));

    

    if (strcmp(argv[4], "max") == 0 ||strcmp(argv[4], "avg") == 0  ) {
        if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " [path_to_bin_file] [Image Size] [in_channel] [Function] [ksize] [stride]" << std::endl;
        return 1;
        }
        int ksize = stoi(argv[5]); 
        int stride = stoi(argv[6]);
        int res = (isize - ksize) / stride + 1;
        float* output = new float[res*res*in_channel];
        float* m_output;
        CHECK_CUDA_ERROR(cudaMalloc(&m_output, res*res*in_channel*sizeof(float)));

        dim3 p1_block(16, 16, 1); 
        dim3 p1_grid((res + p1_block.x - 1) / p1_block.x,(res + p1_block.y - 1) / p1_block.y,in_channel);
        if(strcmp(argv[4], "max") == 0 ){ 
            maxpool_cuda<<<p1_grid, p1_block>>>(k_input, m_output, in_channel, isize, ksize, stride);
            cudaDeviceSynchronize();
            cudaError_t error1 = cudaGetLastError();
            if (error1 != cudaSuccess) {
                std::cerr << "Error during max_pool execution: " << cudaGetErrorString(error1) << std::endl;
            }
            CHECK_CUDA_ERROR(cudaMemcpy(output, m_output, res*res*in_channel * sizeof(float), cudaMemcpyDeviceToHost));
            saveArrayToFile(output, res*res*in_channel, "/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/output/out_gpu.txt");
        }
        else{
            avgpool_cuda<<<p1_grid, p1_block>>>(k_input, m_output, in_channel, isize, ksize, stride);
            cudaDeviceSynchronize();
            cudaError_t error1 = cudaGetLastError();
            if (error1 != cudaSuccess) {
                std::cerr << "Error during avg_pool execution: " << cudaGetErrorString(error1) << std::endl;
            }
            CHECK_CUDA_ERROR(cudaMemcpy(output, m_output, res*res*in_channel * sizeof(float), cudaMemcpyDeviceToHost));
            saveArrayToFile(output, res*res*in_channel, "/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/output/out_gpu.txt");
        }
        delete[] output;
        CHECK_CUDA_ERROR(cudaFree(m_output));
    }
    
    else if (strcmp(argv[4], "relu") == 0 ||strcmp(argv[4], "tanh") == 0  ) {
        if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " [path_to_bin_file] [Image Size] [in_channel] [Function]" << std::endl;
        return 1;
        }
        int res = isize*isize*in_channel;
        int r_block = 1024;
        int r_grid = ( res+ r_block - 1) / r_block;
        
        float* output = new float[res];
        
        if(strcmp(argv[4], "relu") == 0 ){
            
            relu<<<r_grid, r_block>>>(k_input, res);  
            cudaError_t error1 = cudaGetLastError();
                if (error1 != cudaSuccess) {
                    std::cerr << "Error during relu execution: " << cudaGetErrorString(error1) << std::endl;
                } 
            
            
            CHECK_CUDA_ERROR(cudaMemcpy(output, k_input, res * sizeof(float), cudaMemcpyDeviceToHost));
            saveArrayToFile(output, res, "/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/output/out_gpu.txt");
        }
        else{
            
            tanh<<<r_grid, r_block>>>(k_input, res);  
            cudaError_t error1 = cudaGetLastError();
                if (error1 != cudaSuccess) {
                    std::cerr << "Error during tanh execution: " << cudaGetErrorString(error1) << std::endl;
                } 
            
            CHECK_CUDA_ERROR(cudaMemcpy(output, k_input, res * sizeof(float), cudaMemcpyDeviceToHost));
            saveArrayToFile(output, res, "/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/output/out_gpu.txt");
        }
         delete[] output;
         
    }
    
    delete[] input;
   
    CHECK_CUDA_ERROR(cudaFree(d_conv1));
    CHECK_CUDA_ERROR(cudaFree(d_conv2));
    CHECK_CUDA_ERROR(cudaFree(d_fc1));
    CHECK_CUDA_ERROR(cudaFree(d_fc2));
    

    return 0;

}
