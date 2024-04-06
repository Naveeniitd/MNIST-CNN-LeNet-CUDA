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
#define c1_width 520
#define f2_width 5010
struct Weights {
    float* conv2;
    float* fc1;
    float* conv1;
    float* fc2;    
};
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
//----------------------------------------------DEVICE KERNEL CODE----------------------------------------------------------------//


// //---------------------------------------------Convulation without Padding---------------------------------------------------------------------------------//


__global__ void conv_cuda(const float* input, const float* weights, float* output, int in_channel, int out_channel, int isize, int ksize) {
    int res = isize - ksize + 1;

    // Calculate output coordinates (x, y, o)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int o = blockIdx.z * blockDim.z + threadIdx.z;

    if (o < out_channel && x < res && y < res) {
        float sum = 0.0f;

        // Convolution operation
        for (int c = 0; c < in_channel; ++c) {
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    int iy = y + ky;
                    int ix = x + kx;

                    if (ix < isize && iy < isize) {
                        sum += input[c * isize * isize + iy * isize + ix] *
                               weights[o * (in_channel * ksize * ksize) + c * (ksize * ksize) + ky * ksize + kx];
                    }
                }
            }
        }

        output[o * res * res + y * res + x] = sum + weights[out_channel * in_channel * ksize * ksize + o];
    }
}
//---------------------------------------------------------Relu--------------------------------------------------------------------------------//
__global__ void relu(float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = max(0.0f, input[i]);
    }
}
//----------------------------------------------------------Tanh-------------------------------------------------------------------------------//
__global__ void tanh(float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i< n) {
        input[i] = tanhf(input[i]); 
    }
}
//------------------------------------------------KernelmaxPooling-----------------------------------------------------------------------------//
__global__ void maxPooling(const float* input, float* output, int in_channel, int isize, int ksize, int stride) {
    int res = (isize - ksize) / stride + 1;
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < res && y < res && channel < in_channel) {
        const float* currentInput = input + channel * isize * isize;
        float* currentOutput = output + channel * res * res;

        float maxVal = -FLT_MAX;
        for (int ky = 0; ky < ksize; ++ky) {
            for (int kx = 0; kx < ksize; ++kx) {
                int iy = y * stride + ky;
                int ix = x * stride + kx;
                if (ix < isize && iy < isize) {
                    int inputIndex = iy * isize + ix;
                    maxVal = max(maxVal, currentInput[inputIndex]);
                }
            }
        }
        currentOutput[y * res + x] = maxVal;
    }
}
//----------------------------------------fully connnected with relu-------------------------------------------------------------//
__global__ void fconv_cudaR(const float* input, const float* weights, float* output, int in_channel, int out_channel, int isize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Unique thread index
    int inputSize = isize * isize * in_channel; // Total size of the input for one filter

    if (index < out_channel) { // Check if thread index is within the range of output channels
        const float* currentWeights = weights + index * inputSize; // Adjust index for multi-channel
        float bias = weights[out_channel * inputSize + index]; // Bias index adjusted for flattened input
        float sum = 0.0f;

        for (int j = 0; j < inputSize; ++j) {
            sum += input[j] * currentWeights[j];
        }

        // Apply bias and ReLU activation
        output[index] = max(0.0f, sum + bias); // Using max for ReLU
    }
}
//----------------------------------------fully connected without relu--------------------------------------------------//
__global__ void fconv_cuda(const float* input, const float* weights, float* output, int in_channel, int out_channel, int isize) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Unique thread index
    int inputSize = isize * isize * in_channel; // Total size of the input for one filter

    if (index < out_channel) { // Check if thread index is within the range of output channels
        const float* currentWeights = weights + index * inputSize; // Adjust index for multi-channel
        float bias = weights[out_channel * inputSize + index]; // Bias index adjusted for flattened input
        float sum = 0.0f;

        for (int j = 0; j < inputSize; ++j) {
            sum += input[j] * currentWeights[j];
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
//--------------------------------------------------_Softmax_--------------------------------------------------------------------//
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
//---------------------------------------------Find Top 5-----------------------------------------------------------//
void findTop5(const float* softmax_probs, int num_classes, std::vector<int>& top_classes, std::vector<float>& top_probs) {
    // Using a min heap to store the top probabilities and their corresponding indices
    priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> min_heap;

    for (int i = 0; i < num_classes; ++i) {
        float prob = softmax_probs[i];
        if (min_heap.size() < 5) { // If the heap is not full
            min_heap.push({prob, i});
        } else if (prob > min_heap.top().first) { // If current prob is greater than the smallest in the heap
            min_heap.pop(); // Remove the smallest
            min_heap.push({prob, i}); // Insert the current
        }
    }

    // Extract the top 5
    top_classes.clear();
    top_probs.clear();
    while (!min_heap.empty()) {
        top_probs.push_back(min_heap.top().first);
        top_classes.push_back(min_heap.top().second);
        min_heap.pop();
    }

    // Reverse the order to make it from highest to lowest
    std::reverse(top_classes.begin(), top_classes.end());
    std::reverse(top_probs.begin(), top_probs.end());
}
//------------------Function to load binary img in float* input-----------------------//
bool imgload(const string& path, float* input){
    ifstream file(path.c_str(), ios::binary);
    file.read(reinterpret_cast<char*>(input), 28*28*sizeof(float));
    file.close();
    return true;
}

//------------------------------------MAIN FUNCTION----------------------------

int main() {
    string saveDirectory = "/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/output/output_single/";
    string fileExtension = ".txt";
    float correct_output = 0;
    //---------------Reading Trained Weights in Weights struct datatype ----------------------//
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
             
    std::string directory = "/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/pre-proc-img/binary_img/";
    DIR* dir;
    struct dirent* ent;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    if ((dir = opendir(directory.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            std::string filename = ent->d_name;
            
            // Check if the file is a .bin file
            if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".bin") {
                string num = filename.substr(filename.length()-5, 1);

                //cout << num << endl;
                std::string filepath = directory + filename;
                float* input = new float[28 * 28];
                std::ifstream file(filepath, std::ios::binary);
                //std::cout << "Trying to open: " << filepath << std::endl;

                if (!file) {
                    std::cerr << "Cannot open file!" << std::endl;
                    return 1;
                }

                // Read the entire image data into the array
                file.read(reinterpret_cast<char*>(input), 28*28 * sizeof(float));
                if (!file) {
                    std::cerr << "Error reading file or file too short!" << std::endl;
                    return 2;
                }
                file.close();
                float*c1_input;
                CHECK_CUDA_ERROR(cudaMalloc(&c1_input, 28*28*1*sizeof(float)));
                CHECK_CUDA_ERROR(cudaMemcpy(c1_input, input, 28*28*1 * sizeof(float), cudaMemcpyHostToDevice));

                //-------------------Memory alloc in device for different output of layers---------------------------------------------------//
                float *c1_output, *p1_output, *c2_output, *p2_output, *f1_output, *f2_output, *relu_output;
                
                CHECK_CUDA_ERROR(cudaMalloc(&c1_output, 24*24*20*sizeof(float)));
                CHECK_CUDA_ERROR(cudaMalloc(&p1_output, 12*12*20*sizeof(float)));
                CHECK_CUDA_ERROR(cudaMalloc(&c2_output, 8*8*50*sizeof(float)));
                CHECK_CUDA_ERROR(cudaMalloc(&p2_output, 4*4*50*sizeof(float)));
                CHECK_CUDA_ERROR(cudaMalloc(&f1_output, 500*sizeof(float)));
                CHECK_CUDA_ERROR(cudaMalloc(&relu_output, 500*sizeof(float))); 
                CHECK_CUDA_ERROR(cudaMalloc(&f2_output, 10 * sizeof(float)));
                
                //---------LENET architecure with Layer,Input_dim,output_dim,Input_Channels,Output_Channels,Kernel,Stride,Padding,Has Relu ?,No of Weights,Bias,Total Weights-------------------------------------------//
                
                //--------------------------------conv1-28,24,1,20,5,1,0,0,500,20,520--------------------------------------------------------//
                dim3 c1_block(16, 16, 1); 
                dim3 c1_grid((24 + c1_block.x - 1) / c1_block.x,(24 + c1_block.y - 1) / c1_block.y,20);
                conv_cuda<<<c1_grid, c1_block>>>(c1_input, d_conv1, c1_output, 1, 20, 28, 5);
                cudaDeviceSynchronize(); // Wait for the kernel to complete
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    std::cerr << "Error during conv_cuda_1 execution: " << cudaGetErrorString(error) << std::endl;
                }
                //-----------------------------------Pool_1,24,12,20,20,2,2,0,0,-,-,--------------------------------------------------------------//
                dim3 p1_block(16, 16, 1); 
                dim3 p1_grid((12 + p1_block.x - 1) / p1_block.x,(12 + p1_block.y - 1) / p1_block.y,20);
                maxPooling<<<p1_grid, p1_block>>>(c1_output, p1_output, 20, 24, 2, 2);
                cudaDeviceSynchronize();
                cudaError_t error1 = cudaGetLastError();
                if (error1 != cudaSuccess) {
                    std::cerr << "Error during Max_pool_1 execution: " << cudaGetErrorString(error1) << std::endl;
                }
                //------------------------------------Conv_2,12,8,20,50,5,1,0,0,25000,50,25050--------------------------------------------------------//
                dim3 c2_block(8, 8); 
                dim3 c2_grid((8 + c2_block.x - 1) / c2_block.x,(8 + c2_block.y - 1) / c2_block.y,50); 
                conv_cuda<<<c2_grid, c2_block>>>(p1_output, d_conv2, c2_output, 20, 50, 12, 5);
                cudaDeviceSynchronize();
                cudaError_t error2 = cudaGetLastError();
                if (error2 != cudaSuccess) {
                    std::cerr << "Error during conv_cuda_2 execution: " << cudaGetErrorString(error2) << std::endl;
                }
                //------------------------------------------Pool_2,8,4,50,50,2,2,0,0,-,-,-----------------------------------------------//
                dim3 p2_block(4, 4);
                dim3 p2_grid(1, 1, 50);
                maxPooling<<<p2_grid, p2_block>>>(c2_output, p2_output, 50, 8, 2, 2);
                cudaDeviceSynchronize();
                cudaError_t error3 = cudaGetLastError();
                if (error3 != cudaSuccess) {
                    std::cerr << "Error during Max_pool_2 execution: " << cudaGetErrorString(error3) << std::endl;
                }
                //--------------------------------------------FC_1,4,1,50,500,4,1,0,1,400000,500,400500-------------------------------------//
                int f1_block = 256;
                int f1_grid = (500 + f1_block - 1) / f1_block;
                dim3 threads1(f1_block);
                dim3 blocks1(f1_grid);
                fconv_cudaR<<<blocks1, threads1>>>(p2_output, d_fc1, f1_output, 50, 500, 4);
                cudaError_t error4 = cudaGetLastError();
                if (error4 != cudaSuccess) {
                    std::cerr << "Error during fully_conv_cuda_1 execution: " << cudaGetErrorString(error4) << std::endl;
                }
                //----------------------------------------FC_2,1,1,500,10,1,1,0,0,5000,10,5010-----------------------------------------------------// 
                dim3 threads(32);
                dim3 blocks(1);
                fconv_cuda<<<blocks, threads>>>(f1_output, d_fc2, f2_output, 500, 10, 1);
                cudaError_t error5 = cudaGetLastError();
                if (error5 != cudaSuccess) {
                    std::cerr << "Error during fully_conv_cuda_2 execution: " << cudaGetErrorString(error5) << std::endl;
                }
                //-------------------------------------------SOFTMAX PROBABILITIES---------------------------------------------------------------------------//
                float* test6 = new float[10];
                CHECK_CUDA_ERROR(cudaMemcpy(test6, f2_output, 10 * sizeof(float), cudaMemcpyDeviceToHost));
                softmax(test6, 10);     
                vector<int> top_classes;
                vector<float> top_probs;
                findTop5(test6, 10, top_classes, top_probs);
                string savePath = saveDirectory + filename.substr(0, filename.length() - 4) + "_top5" + fileExtension; // Assuming filename has '.bin' extension
                // Open a file stream to write
                std::ofstream outFile(savePath);
                //cout << savePath << endl;
                if (!outFile.is_open()) {
                    std::cerr << "Failed to open file for writing: " << savePath << std::endl;
                    return -1; // or handle the error based on your application's needs
                }
                //cout << filename.substr(10,11) << endl;
                if(top_classes[0]==stoi(num)){
                    correct_output++;
                }
                // Write top 5 probabilities and their classes to the file
                for (size_t i = 0; i < top_classes.size(); ++i) {
                    outFile << "Class " << top_classes[i] << " Probability: " << top_probs[i] << std::endl;
                    
                }
                outFile.close();
                delete[] test6;
                delete[] input;
                CHECK_CUDA_ERROR(cudaFree(c1_input));
                CHECK_CUDA_ERROR(cudaFree(c1_output));
                CHECK_CUDA_ERROR(cudaFree(c2_output));
                CHECK_CUDA_ERROR(cudaFree(p1_output));
                CHECK_CUDA_ERROR(cudaFree(p2_output)); 
                CHECK_CUDA_ERROR(cudaFree(f1_output));
                CHECK_CUDA_ERROR(cudaFree(relu_output));
                CHECK_CUDA_ERROR(cudaFree(f2_output));
            }
            
        }
        closedir(dir);
    } else {
        // Could not open directory
        std::cerr << "Could not open directory" << std::endl;
        return EXIT_FAILURE;
    }  

    cout << "Accuracy of result is " << (correct_output/10000)*100 << endl;
    delete[] weights.conv2;
    delete[] weights.fc1;
    delete[] weights.fc2;
    delete[] weights.conv1;
    CHECK_CUDA_ERROR(cudaFree(d_conv1));
    CHECK_CUDA_ERROR(cudaFree(d_conv2)); 
    CHECK_CUDA_ERROR(cudaFree(d_fc1)); 
    CHECK_CUDA_ERROR(cudaFree(d_fc2));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Time taken: %f ms\n", milliseconds);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop)); 
    return 0;
}

