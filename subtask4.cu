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
// __constant__ float d_conv1[c1_width];
// __constant__ float d_fc2[f2_width];
struct Weights {
    //float conv1[520];
    float* conv2;
    float* fc1;
    //float fc2[5010];
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

void printVector(const vector<float>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}


//------------------------------DEVICE KERNEL CODE-------------------------------------//


// //---------------------------Convulation without Padding---------------------------//


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
// __global__ void relu(float* input, float* output, int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) {
//         output[i] = max(0.0f, input[i]);
//     }
// }

// __global__ void tanh(float* input, int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i< n) {
//         input[i] = tanhf(input[i]); 
//     }
// }
//------------------------------------------------KernelMaxPooling-----------------------------------------------------------------------------//
__global__ void MaxPooling(const float* input, float* output, int in_channel, int isize, int ksize, int stride) {
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
//------------------------------------HOST-CODE-------------------------------------------------------------------//
//-------------------------------Max Pooling-----------------------------------------------------------------------//
void maxPooling(const float* input, float* output, int in_channel, int isize, int ksize, int stride) {
    // Calculate the dimension of the output feature maps
    int res = (isize - ksize) / stride + 1;

    // Iterate over each input channel
    for (int channel = 0; channel < in_channel; ++channel) {
        // Calculate the starting pointers for the current channel in the input and output
        const float* currentInput = input + channel * isize * isize;
        float* currentOutput = output + channel * res * res;

        // Apply max pooling for the current channel
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                float maxVal = -FLT_MAX; // Initialize to the smallest float value
                for (int ky = 0; ky < ksize; ++ky) {
                    for (int kx = 0; kx < ksize; ++kx) {
                        int iy = y * stride + ky; // Calculate the input's y-coordinate
                        int ix = x * stride + kx; // Calculate the input's x-coordinate
                        // Check if the coordinates are within the bounds of the input size
                        if (ix < isize && iy < isize) {
                            int inputIndex = iy * isize + ix; // Calculate the index in the flattened input
                            maxVal = std::max(maxVal, currentInput[inputIndex]);
                        }
                    }
                }
                currentOutput[y * res + x] = maxVal; // Assign the max value to the output
            }
        }
    }
}
//---------------------------------Convolution without Padding-------------------------------------------------------------//
void conv(const float* input, const float* weights, float* output, int in_channel, int out_channel, int isize, int ksize) {
    
    int res = isize - ksize + 1;

    // Iterate over each output channel
    for(int o = 0; o < out_channel; ++o) {
        // Iterate over the output spatial dimension
        for(int y = 0; y < res; ++y) {
            for(int x = 0; x < res; ++x) {
                float sum = 0.0f; // Accumulator for the sum
                
                // Iterate over each input channel
                for(int c = 0; c < in_channel; ++c) {
                    // Convolve the kernel with the input
                    for(int ky = 0; ky < ksize; ++ky) {
                        for(int kx = 0; kx < ksize; ++kx) {
                            int iy = y + ky; // Calculate the input y-coordinate
                            int ix = x + kx; // Calculate the input x-coordinate
                            
                            // Ensure the coordinates are within the bounds of the input dimensions
                            if (ix < isize && iy < isize) {
                            
                                sum += input[c * isize * isize + iy * isize + ix] * weights[o * (in_channel * ksize * ksize) + c * (ksize * ksize) + ky * ksize + kx];
                            }
                        }
                    }
                }
            
                output[o * res * res + y * res+ x] = sum + weights[out_channel * in_channel * ksize * ksize + o];
            }
        }
    }
}


float sigmoid(float x){
    return 1.0f/(1.0f+exp(-x));
}

vector<float> sigfunc(const vector<float>& input){
    vector<float> output(input.size());
    transform(input.begin(), input.end(), output.begin(), sigmoid);
    return output;
}

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


//----------------------Function to load the batch of images into host memory----------------------------------------------------//
float* loadBatch(const char* filename, int& dataSize) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return nullptr;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    float* buffer = new float[size / sizeof(float)];
    if (!file.read(reinterpret_cast<char*>(buffer), size)) {
        std::cerr << "Failed to read file: " << filename << std::endl;
        delete[] buffer;
        return nullptr;
    }

    dataSize = size / sizeof(float); // Total number of float elements read
    return buffer;
}

//----------------------------------------fully connected Convolution---------------------------------------------------//
void fconv(const float* input, const float* weights, float* output, int in_channel, int out_channel, int isize) {
    // For FC_1, the output dimension is always 1 per filter
    
    int inputSize = isize * isize * in_channel; // Total size of the input for one filter
    
    for (int i = 0; i < out_channel; ++i) {
        const float* currentWeights = weights + i * inputSize; // Adjust index for multi-channel
        float bias = weights[out_channel * inputSize + i]; // Bias index adjusted for flattened input
        float sum = 0.0f;
        
        for (int j = 0; j < inputSize; ++j) {
            sum += input[j] * currentWeights[j];
        }
        
        // Apply bias and ReLU activation
        output[i] = std::max(0.0f, sum + bias);

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
//--------------------------Function-to-read-trained-weights---------------------------//
float* fileReadP(const string& path, int size) {
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
bool fileRead(const string& path, float* weights, int size) {
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << path << endl;
        return false;
    }
    for (int i = 0; i < size; i++) {
        if (!(file >> weights[i])) {
            cerr << "Failed to read weight at position " << i << " from file: " << path << endl;
            return false;
        }
    }
    file.close();
    return true;
}
//------------------------------------relu-------------------------------------------------------------//
void relu(float* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = std::max(0.0f, input[i]);
    }
}

//------------------Function to load binary img in float* input-----------------------//
bool imgload(const string& path, float* input){
    ifstream file(path.c_str(), ios::binary);
    file.read(reinterpret_cast<char*>(input), 28*28*sizeof(float));
    file.close();
    return true;
}
//-------------------Function to load batch binary img in float* input-----------------------------------//
bool batchImgLoad(const string& path, float* input, int batch_size) {
    ifstream file(path.c_str(), ios::binary);
    if (!file) {
        cerr << "Could not open the file: " << path << endl;
        return false;
    }
    file.read(reinterpret_cast<char*>(input), 28*28*sizeof(float)*batch_size);
    file.close();
    return true;
}
//------------------------------------MAIN FUNCTION----------------------------

int main() {
    const int batch_size = 100;
    const int isize = 28 * 28;
    const int cuda_stream = 4;

    float* host_imgbatch[cuda_stream];
    float* gpu_imgbatch[cuda_stream];
    for (int i = 0; i < cuda_stream; ++i) {
        cudaMallocHost(&host_imgbatch[i], isize * batch_size * sizeof(float)); // Pinned memory
        cudaMalloc(&gpu_imgbatch[i], isize * batch_size * sizeof(float));
    }
    cudaStream_t cuda_streams[cuda_stream];
    for (int i = 0; i < cuda_stream; ++i) {
        cudaStreamCreate(&cuda_streams[i]);
    }

    //---------------Reading Trained Weights in Weights struct datatype ----------------------//
    Weights weights;
    // if (!fileReadP("trained_weights/conv1.txt", weights.conv1, 520)) {
    //     // Handle error
    //     return -1;}
    weights.conv1 = fileReadP("trained_weights/conv1.txt", 520);
    weights.conv2 = fileReadP("trained_weights/conv2.txt", 25050);
    weights.fc1 = fileReadP("trained_weights/fc1.txt", 400500);
    weights.fc2 = fileReadP("trained_weights/fc2.txt", 5010);
    // if (!fileReadP("trained_weights/fc2.txt", weights.fc2, 5010)) {
    //     // Handle error
    //     return -1;}
        //printArray(weights.conv1, 520);
        //printArray(weights.fc1, 400500);
        //printArray(weights.conv2, 25050);
    //printArray(weights.fc2, 5010);
    float *d_conv1, *d_fc2, *d_conv2, *d_fc1; //variable for Device holding pointer to the data of conv1.txt, conv2.txt fc1.txt and fc2.txt respectively
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv1, 520 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv1, weights.conv1, 520 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv2, 25050 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv2, weights.conv2, 25050 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc1, 400500 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc1, weights.fc1, 400500 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc2, 5010 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc2, weights.fc2, 5010 * sizeof(float), cudaMemcpyHostToDevice));
             
    
    std::string directory = "batch_binary/";
    DIR* dir;
    struct dirent* ent;
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    if ((dir = opendir(directory.c_str())) != nullptr) {
        int batch_index = 0;
        while ((ent = readdir(dir)) != nullptr && batch_index < 100) {
            int stream_idx = batch_index % cuda_stream;
            std::string filename = ent->d_name;

            // Check if the file is a .bin file
            if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".bin") {
                std::string filepath = directory + filename;
                batchImgLoad(filepath,host_imgbatch[batch_index], batch_size );
                cudaMemcpyAsync(gpu_imgbatch[stream_idx], host_imgbatch[stream_idx], isize * batch_size * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[stream_idx]);
                //-------------------Memory alloc in device for different output of layers---------------------------------------------------//
                float *c1_output, *p1_output, *c2_output, *p2_output, *f1_output, *f2_output, *relu_output;
                
                CHECK_CUDA_ERROR(cudaMalloc(&c1_output, 24*24*20*sizeof(float)));
                CHECK_CUDA_ERROR(cudaMalloc(&p1_output, 12*12*20*sizeof(float)));
                CHECK_CUDA_ERROR(cudaMalloc(&c2_output, 8*8*50*sizeof(float)));
                CHECK_CUDA_ERROR(cudaMalloc(&p2_output, 4*4*50*sizeof(float)));
                CHECK_CUDA_ERROR(cudaMalloc(&f1_output, 500*sizeof(float)));
                CHECK_CUDA_ERROR(cudaMalloc(&relu_output, 500*sizeof(float))); 
                CHECK_CUDA_ERROR(cudaMalloc(&f2_output, 10 * sizeof(float)));
                for (int imgIdx = 0; imgIdx < batch_size; imgIdx++) {
                    int in_channel = 1;
                    int ksize =5;
                    int out_channel = 20;

                    // Calculate pointers for the current image in the batch
                    const float* cur_input = gpu_imgbatch + imgIdx * in_channel * isize * isize;
                    float* cur_output = gpu_imgbatch + imgIdx * out_channel * (isize - ksize + 1) * (isize - ksize + 1);

                    // float*c1_input;
                    // CHECK_CUDA_ERROR(cudaMalloc(&c1_input, 28*28*1*batch_size*sizeof(float)));
                    // CHECK_CUDA_ERROR(cudaMemcpy(c1_input, input, 28*28*1 *batch_size* sizeof(float), cudaMemcpyHostToDevice));
                    //-------------------------------Host_to_Device_Copy_of_Trained_Weights---------------------------------------//
                    
                    
                //---------LENET architecure with Layer,Input_dim,output_dim,Input_Channels,Output_Channels,Kernel,Stride,Padding,Has Relu ?,No of Weights,Bias,Total Weights-------------------------------------------//
                    
                //---------------------------------------------------------------------------------------------------------------------//
                        
                        //printArray(input, 28*28);
                //--------------------------------conv1-28,24,1,20,5,1,0,0,500,20,520--------------------------------------------------------//
                    dim3 c1_block(16, 16, 1); // Example thread block size
                    dim3 c1_grid((24 + c1_block.x - 1) / c1_block.x,
                            (24 + c1_block.y - 1) / c1_block.y,
                            20);


                    conv_cuda<<<c1_grid, c1_block>>>(c1_input, d_conv1, c1_output, 1, 20, 28, 5);
                    cudaDeviceSynchronize(); // Wait for the kernel to complete
                    cudaError_t error = cudaGetLastError();
                    if (error != cudaSuccess) {
                        std::cerr << "Error during conv_cuda_1 execution: " << cudaGetErrorString(error) << std::endl;
                    }
            //-----------------------------------Pool_1,24,12,20,20,2,2,0,0,-,-,--------------------------------------------------------------//
                    // float* test_output = new float[24*24*20];
                    // cudaMemcpy(test_output, c1_output, 24*24*20 * sizeof(float), cudaMemcpyDeviceToHost);

                    // //cout << weights.conv1.data() << " ";
                    // //printArray(test_output, 24*24*20 );
                    // float* test2 = new float[12*12*20];
                    // maxPooling(test_output, test2 , 20, 24, 2, 2);
                    dim3 p1_block(16, 16, 1); // Example thread block size, may need adjustment
                    dim3 p1_grid((12 + p1_block.x - 1) / p1_block.x,
                            (12 + p1_block.y - 1) / p1_block.y,
                            20);

                    MaxPooling<<<p1_grid, p1_block>>>(c1_output, p1_output, 20, 24, 2, 2);
                    cudaDeviceSynchronize();
                    cudaError_t error1 = cudaGetLastError();
                    if (error1 != cudaSuccess) {
                        std::cerr << "Error during Max_pool_1 execution: " << cudaGetErrorString(error1) << std::endl;
                    }
            //------------------------------------Conv_2,12,8,20,50,5,1,0,0,25000,50,25050--------------------------------------------------------//
                    //printArray(test2,12*12*20 );
                    // float* test3 = new float[8*8*50];
                    // float* test_output = new float[12*12*20];
                    // cudaMemcpy(test_output, p1_output, 12*12*20 * sizeof(float), cudaMemcpyDeviceToHost);
                    // conv(test_output,  weights.conv2, test3, 20, 50, 12, 5);
                    //printArray(test3, 8*8*50);


                    dim3 c2_block(8, 8); // One thread per output pixel
                    dim3 c2_grid((8 + c2_block.x - 1) / c2_block.x,
                            (8 + c2_block.y - 1) / c2_block.y,
                            50); // 50 channels to process

                    conv_cuda<<<c2_grid, c2_block>>>(p1_output, d_conv2, c2_output, 20, 50, 12, 5);
                    cudaDeviceSynchronize();
                    cudaError_t error2 = cudaGetLastError();
                    if (error2 != cudaSuccess) {
                        std::cerr << "Error during conv_cuda_2 execution: " << cudaGetErrorString(error2) << std::endl;
                    }
                //------------------------------------------Pool_2,8,4,50,50,2,2,0,0,-,-,-----------------------------------------------//
                    
                    dim3 p2_block(4, 4);
                    dim3 p2_grid(1, 1, 50);
                    MaxPooling<<<p2_grid, p2_block>>>(c2_output, p2_output, 50, 8, 2, 2);
                    cudaDeviceSynchronize();
                    cudaError_t error3 = cudaGetLastError();
                    if (error3 != cudaSuccess) {
                        std::cerr << "Error during Max_pool_2 execution: " << cudaGetErrorString(error3) << std::endl;
                    }
                    // float* test4 = new float[4*4*50];
                    // float* test_output = new float[8*8*50];
                    // cudaMemcpy(test_output, c2_output, 8*8*50 * sizeof(float), cudaMemcpyDeviceToHost);
                    // maxPooling(test_output, test4 , 50, 8, 2, 2);
                    //printArray(test4, 4*4*50);


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
                    // float* test5 = new float[500];
                    // float* test_output = new float[4*4*50];
                    // cudaMemcpy(test_output, p2_output, 4*4*50 * sizeof(float), cudaMemcpyDeviceToHost);
                    // fconv(test_output, weights.fc1, test5, 50, 500, 4);
                    //printArray(test5, 500);
                    // relu(test5, 500);
                    // float* test6 = new float[10];
                    // fconv(test5, weights.fc2, test6, 500, 10, 1);
                //----------------------------------------FC_2,1,1,500,10,1,1,0,0,5000,10,5010-----------------------------------------------------//
                    // printArray(test6, 10);
                
                    dim3 threads(32);
                    dim3 blocks(1);
                    fconv_cuda<<<blocks, threads>>>(f1_output, d_fc2, f2_output, 500, 10, 1);
                    cudaError_t error5 = cudaGetLastError();
                    if (error5 != cudaSuccess) {
                        std::cerr << "Error during fully_conv_cuda_2 execution: " << cudaGetErrorString(error5) << std::endl;
                    }
                    float* test6 = new float[10];
                    CHECK_CUDA_ERROR(cudaMemcpy(test6, f2_output, 10 * sizeof(float), cudaMemcpyDeviceToHost));
                    softmax(test6, 10);
                    // printArray(test6, 10);
                    vector<int> top_classes;
                    vector<float> top_probs;
                    findTop5(test6, 10, top_classes, top_probs);


                    //Print top 5 probabilities and their classes
                    for (size_t i = 0; i < top_classes.size(); ++i) {
                        std::cout << "Class " << top_classes[i] << " Probability: " << top_probs[i] << std::endl;
                    }

                    
                    //delete[] test_output;
                    //delete[] test2;
                    //delete[] test3;
                    //delete[] test4;
                    //delete[] test5;
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
                    // float test[520];
                    // CHECK_CUDA_ERROR(cudaMemcpyFromSymbol(test, d_fc2, 5010 * sizeof(float)));
                    //printArray(test, 520);
                }
            }
            
            closedir(dir);
        }    
    }             
    else {
        // Could not open directory
        std::cerr << "Could not open directory" << std::endl;
        return EXIT_FAILURE;
    }     
    for (int i = 0; i < cuda_stream; i++) {
        cudaStreamSynchronize(cuda_streams[i]);
    }
    for (int i = 0; i < cuda_stream; i++) {
        cudaStreamDestroy(cuda_streams[i]);
    }
    delete[] weights.conv2;
    delete[] weights.fc1;
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
