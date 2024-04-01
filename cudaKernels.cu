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
#include <iomanip>
#include <utility>
#include <queue>
using namespace std;

struct Weights {
    float* conv1;
    float* conv2;
    float* fc1;
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

//

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
    int res = 1; // Not used in the same way as in convolution
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

//------------------------------------MAIN FUNCTION-----------------------------------------//
int main() {
    //---------------Reading Trained Weights in Weights struct datatype ----------------------//
    Weights weights;
    weights.conv1 = fileRead("trained_weights/conv1.txt", 520);
    weights.conv2 = fileRead("trained_weights/conv2.txt", 25050);
    weights.fc1 = fileRead("trained_weights/fc1.txt", 400500);
    weights.fc2 = fileRead("trained_weights/fc2.txt", 5010);
    // printArray(weights.conv1, 520);
    //-----------------Reading Image---------------------------------------------------//
    float* input = new float[28 * 28];
    std::ifstream file("2.bin", std::ios::binary);
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
    cudaMalloc(&c1_input, 28*28*1*sizeof(float));
    cudaMemcpy(c1_input, input, 28*28*1 * sizeof(float), cudaMemcpyHostToDevice);
    //-------------------------------Host_to_Device_Copy_of_Trained_Weights---------------------------------------//

    float *d_conv1, *d_conv2, *d_fc1, *d_fc2; //variable for Device holding pointer to the data of conv1.txt, conv2.txt fc1.txt and fc2.txt respectively
    cudaMalloc(&d_conv1, 520 * sizeof(float));
    cudaMemcpy(d_conv1, weights.conv1, 520 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_conv2, 25050 * sizeof(float));
    cudaMemcpy(d_conv2, weights.conv2, 25050 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_fc1, 400500 * sizeof(float));
    cudaMemcpy(d_fc1, weights.fc1, 400500 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_fc2, 5010 * sizeof(float));
    cudaMemcpy(d_fc2, weights.fc2, 5010 * sizeof(float), cudaMemcpyHostToDevice);
    

    //-------------------Memory alloc in device for different output of layers---------------------------------------------------//
    float *c1_output, *p1_output, *c2_output, *p2_output, *f1_output, *f2_output, *relu_output;
    
    cudaMalloc(&c1_output, 24*24*20*sizeof(float));
    cudaMalloc(&p1_output, 12*12*20*sizeof(float));
    cudaMalloc(&c2_output, 8*8*50*sizeof(float));
    cudaMalloc(&p2_output, 4*4*50*sizeof(float));
    cudaMalloc(&f1_output, 500*sizeof(float));
    cudaMalloc(&relu_output, 500*sizeof(float)); 
    cudaMalloc(&f2_output, 10 * sizeof(float));
    
    //---------LENET architecure with Layer,Input_dim,output_dim,Input_Channels,Output_Channels,Kernel,Stride,Padding,Has Relu ?,No of Weights,Bias,Total Weights-------------------------------------------//
    //const string directory = "Binary_img";
    //for(int i = 0 ; i < 10000; i++){
    // std::stringstream ss;
    // ss << std::setw(6) << std::setfill('0') << i << "-numX.bin";
    // std::string filepath = directory + ss.str();
    // float* input = new float[28 * 28];
    // imgload(filepath, input);       
    // float* d_image;
    // cudaMalloc(&d_image, 28*28 * sizeof(float));
    // cudaMemcpy(d_image, input, 28*28 * sizeof(float), cudaMemcpyHostToDevice);
    // printMatrix(outputMatrix);
//---------------------------------------------------------------------------------------------------------------------//
    
    //printArray(input, 28*28);
//--------------------------------conv1-28,24,1,20,5,1,0,0,500,20,520--------------------------------------------------------//
    dim3 threadsPerBlock(16, 16, 1); // Example thread block size
    dim3 numBlocks((24 + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (24 + threadsPerBlock.y - 1) / threadsPerBlock.y,
               20);

    conv_cuda<<<numBlocks, threadsPerBlock>>>(c1_input, d_conv1, c1_output, 1, 20, 28, 5);
    cudaDeviceSynchronize(); // Wait for the kernel to complete

    // float* test_output = new float[24*24*20];
    // cudaMemcpy(test_output, c1_output, 24*24*20 * sizeof(float), cudaMemcpyDeviceToHost);

    // //cout << weights.conv1.data() << " ";
    // //printArray(test_output, 24*24*20 );
    // float* test2 = new float[12*12*20];
    // maxPooling(test_output, test2 , 20, 24, 2, 2);
    dim3 threadsPerBlock1(16, 16, 1); // Example thread block size, may need adjustment
    dim3 numBlocks1((12 + threadsPerBlock1.x - 1) / threadsPerBlock1.x,
               (12 + threadsPerBlock1.y - 1) / threadsPerBlock1.y,
               20);

    MaxPooling<<<numBlocks1, threadsPerBlock1>>>(c1_output, p1_output, 20, 24, 2, 2);
    cudaDeviceSynchronize();

    //printArray(test2,12*12*20 );
    float* test3 = new float[8*8*50];
    float* test_output = new float[12*12*20];
    cudaMemcpy(test_output, p1_output, 12*12*20 * sizeof(float), cudaMemcpyDeviceToHost);
    conv(test_output,  weights.conv2, test3, 20, 50, 12, 5);
    //printArray(test3, 8*8*50);






    float* test4 = new float[4*4*50];
    maxPooling(test3, test4 , 50, 8, 2, 2);
    //printArray(test4, 4*4*50);
    float* test5 = new float[500];
    fconv(test4, weights.fc1, test5, 50, 500, 4);
    //printArray(test5, 500);
    relu(test5, 500);
    float* test6 = new float[10];
    fconv(test5, weights.fc2, test6, 500, 10, 1);
    // printArray(test6, 10);
    softmax(test6, 10);
    // printArray(test6, 10);
    vector<int> top_classes;
    vector<float> top_probs;
    findTop5(test6, 10, top_classes, top_probs);


    // Print top 5 probabilities and their classes
    for (size_t i = 0; i < top_classes.size(); ++i) {
        std::cout << "Class " << top_classes[i] << " Probability: " << top_probs[i] << std::endl;
    }

    delete[] weights.conv1;
    delete[] weights.conv2;
    delete[] weights.fc1;
    delete[] weights.fc2;
    delete[] test_output;
    //delete[] test2;
    delete[] test3;
    delete[] test4;
    delete[] test5;
    delete[] test6;
    delete[] input;
    cudaFree(c1_input);
    cudaFree(c1_output);
    // cudaFree(c2_output);
    // cudaFree(p1_output);
    // cudaFree(p2_output);
    // cudaFree(d_conv1);
    // cudaFree(d_conv2); 
    // cudaFree(d_fc1); 
    // cudaFree(d_fc2);
    // cudaFree(f1_output);
    // cudaFree(relu_output);
    // cudaFree(f2_output);

    return 0;
}

