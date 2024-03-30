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
using namespace std;

struct Weights {
    vector<float> conv1;
    vector<float> conv2;
    vector<float> fc1;
    vector<float> fc2;
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


//---------------------------Convulation without Padding---------------------------//
__global__ void conv(const float* input, const float* weights, float* output, int isize, int ksize, int out_channel ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int res = isize - ksize + 1;

    if (i < res && j < res && k < out_channel) {
        float sum = 0.0f;
        for (int m = 0; m < ksize; ++m) {
            for (int n = 0; n < ksize; ++n) {
                int X = i + m;
                int Y = j + n;
                sum += input[Y * isize + X] * weights[ k*(ksize*ksize) + ( m * ksize + n)];
            }
        }
        sum+= weights[out_channel*ksize*ksize +k];
        output[k*(res*res) + j * res + i] = sum;
    }
}



__global__ void convpad(const float* input, const float* weights, float* output, int isize, int ksize, int out_channel, int pad){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int res = isize;

    if (i < isize && j < isize && k < out_channel) {
        float sum = 0.0f;
        for (int m = 0; m < ksize; ++m) {
            for (int n = 0; n < ksize; ++n) {
                int x_index = i + m - pad;
                int y_index = j + n - pad;

                // Check boundaries for padding
                if (x_index >= 0 && x_index < isize && y_index >= 0 && y_index < isize) {
                    sum += input[y_index * isize + x_index] * weights[k * (ksize * ksize) + (m * ksize + n)];
                } // Implicit else: sum += 0 for padding areas
            }
        }
        // Add the bias term
        sum += weights[out_channel * ksize * ksize + k];

        output[k * (res * res) + j * res + i] = sum;
    }
}

__global__ void relu(float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = max(0.0f, input[i]);
    }
}

__global__ void tanh(float* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i< n) {
        input[i] = tanhf(input[i]); 
    }
}

//-----------------------------------Max Pooling--------------------------------------------//
__global__ void maxPool(const float* input, float* output, int isize, int output_channel, int poolSize, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Calculate the effective output size considering the stride.
    // This should ideally be done outside the kernel and passed as parameters, but for illustration:
    int osize = (isize - poolSize) / stride + 1; // Correct way to calculate output size

    if (x < osize && y < osize && z < output_channel) {
        float maxVal = -FLT_MAX;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int X = x * stride + j; // Corrected to use 'x' and 'y' within the reduced output size context
                int Y = y * stride + i;
                if (X < isize && Y < isize) {
                    int index = z * isize * isize + Y * isize + X;
                    maxVal = fmaxf(maxVal, input[index]);
                }
            }
        }
        // Correct output indexing considering the new output size calculation
        int output_index = z * osize * osize + y * osize + x;
        output[output_index] = maxVal;
    }
}


//------------------------HOST-CODE-------------------------------//
void maxPoolCPU(const float* input, float* output, int isize, int output_channel, int poolSize, int stride) {
    for (int z = 0; z < output_channel; ++z) {
        for (int y = 0; y < isize; y += stride) {
            for (int x = 0; x < isize; x += stride) {
                float maxVal = -FLT_MAX;
                for (int i = 0; i < poolSize; ++i) {
                    for (int j = 0; j < poolSize; ++j) {
                        int X = x + j;
                        int Y = y + i;
                        if (X < isize && Y < isize) {
                            int index = z * isize * isize + Y * isize + X;
                            maxVal = fmaxf(maxVal, input[index]);
                        }
                    }
                }
                // Calculate the output index considering the stride.
                // We're effectively down-sampling the input matrix based on the stride.
                int ox = x / stride;
                int oy = y / stride;
                int output_index = z * (isize / stride) * (isize / stride) + oy * (isize / stride) + ox;
                output[output_index] = maxVal;
            }
        }
    }
}

void convHost(const float* input, const float* weights, float* output, int isize, int ksize, int out_channel) {
    int res = isize - ksize + 1; // The size of the output feature map

    // Iterate over each output channel
    for (int k = 0; k < out_channel; ++k) {
        // Iterate over the output spatial dimensions
        for (int i = 0; i < res; ++i) {
            for (int j = 0; j < res; ++j) {
                float sum = 0.0f;

                // Perform the element-wise multiplication and sum
                for (int m = 0; m < ksize; ++m) {
                    for (int n = 0; n < ksize; ++n) {
                        int X = i + m; // Corresponding x index in the input
                        int Y = j + n; // Corresponding y index in the input
                        sum += input[Y * isize + X] * weights[k * ksize * ksize + m * ksize + n];
                    }
                }

                // Add the bias term (assuming the bias is stored at the end of weights for each channel)
                sum += weights[out_channel * ksize * ksize + k];

                // Store the result in the output array
                output[k * res * res + j * res + i] = sum;
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

// Function to apply the softmax operation on a float array
void softmax(float* input, float* output, int n) {
    // Find the maximum value in the input array for numerical stability
    float maxVal = -std::numeric_limits<float>::max();
    for (int i = 0; i < n; ++i) {
        if (input[i] > maxVal) {
            maxVal = input[i];
        }
    }

    // Compute the sum of the exponentials of the inputs
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - maxVal); // Subtract maxVal for numerical stability
        sum += output[i];
    }

    // Normalize the output to get probabilities
    for (int i = 0; i < n; ++i) {
        output[i] /= sum;
    }
}


// Function to load the batch of images into host memory
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
//--------------------------Function-to-read-trained-weights---------------------------//
vector<float> fileRead( const string& path) {
    vector<float> weights;
    ifstream file(path.c_str());
    float weight;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return weights;
    }

    while (file >> weight) {
        weights.push_back(weight);
    }

    file.close();
    return weights;
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
    cout<<"start of main function" <<endl;
    Weights weights;
    weights.conv1 = fileRead("trained_weights/conv1.txt");
    weights.conv2 = fileRead("trained_weights/conv2.txt");
    weights.fc1 = fileRead("trained_weights/fc1.txt");
    weights.fc2 = fileRead("trained_weights/fc2.txt");
    // cout << weights.conv1.size()<<" ";
    // cout << weights.conv2.size()<<" ";
    // cout << weights.fc1.size()<< " ";
    // cout << weights.fc2.size()<< " ";
    //printVector(weights.conv1);

    //-----------------------Host_to_Device_Copy_of_Trained_Weights---------------------------------------//

    float *d_conv1, *d_conv2, *d_fc1, *d_fc2; //variable for Device holding pointer to the data of conv1.txt, conv2.txt fc1.txt and fc2.txt respectively
    cudaMalloc(&d_conv1, weights.conv1.size() * sizeof(float));
    cudaMemcpy(d_conv1, weights.conv1.data(), weights.conv1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_conv2, weights.conv2.size() * sizeof(float));
    cudaMemcpy(d_conv2, weights.conv2.data(), weights.conv2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_fc1, weights.fc1.size() * sizeof(float));
    cudaMemcpy(d_fc1, weights.fc1.data(), weights.fc1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_fc2, weights.fc2.size() * sizeof(float));
    cudaMemcpy(d_fc2, weights.fc2.data(), weights.fc2.size() * sizeof(float), cudaMemcpyHostToDevice);
    //------------------imgload from Binary_img folder---------------------------------------------------//
    //cout<<"this done 1"<<endl;
    // float* d_con1test = new float[weights.conv1.size()];
    // cudaMemcpy(d_con1test, d_conv1, weights.conv1.size() * sizeof(float), cudaMemcpyDeviceToHost);
    // printArray(d_con1test, 520);
    //cout <<"this done 2";
    float* input = new float[28 * 28];
    imgload("2.bin", input);
    //delete[] d_con1test;
    //printArray(input, 28*28);

    //-------------------Host to device copy of input/output---------------------------------------------------//
    float*c1_input, *c1_output, *p1_output, *c2_output, *p2_output, *f1_output, *f2_output, *relu_output;
    cudaMalloc(&c1_input, 28*28*1*sizeof(float));
    cudaMemcpy(c1_input, input, 28*28*1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&c1_output, 24*24*20*sizeof(float));
    cudaMalloc(&p1_output, 12*12*20*sizeof(float));
    cudaMalloc(&c2_output, 8*8*50*sizeof(float));
    cudaMalloc(&p2_output, 4*4*50*sizeof(float));
    cudaMalloc(&f1_output, 500*sizeof(float));
    cudaMalloc(&relu_output, 500 * sizeof(float)); 
    cudaMalloc(&f2_output, 10 * sizeof(float));
    // float* d_test = new float[24*24*20];
    // cudaMemcpy(d_test, c1_output, 24*24*20 * sizeof(float), cudaMemcpyDeviceToHost);
    //printArray(d_test, 24*24*20);
    
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

        //----------------------------Conv_1 28,24,1,20,5,1,0,0,500,20,520-----------------------------//

        dim3 c1_block(16, 16, 1); // Block size of 16x16 for spatial dimensions and 1 for channels
        dim3 c1_grid((24 + c1_block.x - 1) / c1_block.x, (24 + c1_block.y - 1) / c1_block.y, 24);


        conv<<<c1_grid, c1_block>>>(c1_input, d_conv1, c1_output, 28, 5, 20);

        // cudaError_t error = cudaGetLastError();
        // if (error != cudaSuccess) {
        //     std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        // }
        // else{
        //     std::cout << error << std::endl;
        // }
        cudaDeviceSynchronize();
        //float* test1 = new float[24*24*20];
        //float* test_output = new float[24*24*20];
        //cudaMemcpy(test1, c1_output, 24*24*20*sizeof(float), cudaMemcpyDeviceToHost);

        //printArray(test1, 24*24*20);
        //convHost(input, weights.conv1.data(), test_output, 28, 5, 20);
        //printArray(test_output, 24*24*20);
        //----------------------------Pool_1,24,12,20,20,2,2,0,0,-,-,- -----------------------------------//
        dim3 p1_block(16, 16, 1);
        dim3 p1_grid((12 + p1_block.x - 1) / p1_block.x, (12 + p1_block.y - 1) / p1_block.y, 20);

        maxPool<<<p1_grid, p1_block>>>(c1_output, p1_output, 24, 20, 2, 2);
        cudaDeviceSynchronize();
        // float* test2 = new float[12*12*20];
        // cudaMemcpy(test2, p1_output, 12*12*20 * sizeof(float), cudaMemcpyDeviceToHost);
        // printArray(test, 12*12*20);
        // maxPoolCPU(test1, test, 24,20,2,2);
        // printArray(test,12*12*20);
        //----------------------------Conv_2,12,8,20,50,5,1,0,0,25000,50,25050-------------------------------------------//
        dim3 c2_block(8, 8, 1);
        dim3 c2_grid(1,1,50);
        
        conv<<<c2_grid, c2_block>>>(p1_output, d_conv2, c2_output, 12, 5, 50);
        cudaDeviceSynchronize();
        float* test3 = new float[8*8*50];
        // //float* test_output = new float[24*24*20];
        cudaMemcpy(test3, c2_output, 8*8*50*sizeof(float), cudaMemcpyDeviceToHost);

        // printArray(test3, 8*8*50);
        // convHost(test2, weights.conv2.data(), test3, 12, 5, 50);
        // printArray(test3, 8*8*50);
        //----------------------------Pool_2,8,4,50,50,2,2,0,0,-,-,- ----------------------------------------------//
        dim3 p2_block(8, 8, 1);
        dim3 p2_grid(1,1, 50);

        maxPool<<<p2_grid, p2_block>>>(c2_output, p2_output, 8, 50, 2, 2);
        
        float* test2 = new float[4*4*50];
        //cudaMemcpy(test2, p2_output, 4*4*50 * sizeof(float), cudaMemcpyDeviceToHost);
        // printArray(test2, 4*4*50);
        maxPoolCPU(test3,test2 , 8,50,2,2);
        printArray(test2,4*4*50);
        
        //------------------------------FC_1,4,1,50,500,4,1,0,1,400000,500,400500---------------------------------------//
        // dim3 f1_block(1,1,500);
        // dim3 f1_grid(1,1,1);
        // conv<<<f1_grid, f1_block>>>(p2_output, d_fc1, f1_output, 4, 4, 500);
        // float* test5 = new float[500];
        // cudaMemcpy(test5, f1_output, 500 * sizeof(float), cudaMemcpyDeviceToHost);
        // printArray(test5, 500);

        // float* test3 = new float[4*4*50];

        // cudaMemcpy(test3, p2_output, 4*4*50*sizeof(float), cudaMemcpyDeviceToHost);
        // convHost(test3, weights.fc1.data(), test5, 4, 4, 500);
        // printArray(test5, 500);
        // //--------------------------------------------RELU-------------------------------------------------------------------//

        // dim3 re_block(256); // Number of threads in a block (tune this based on your GPU)
        // dim3 re_grid((500 + re_block.x - 1) / re_block.x); 
        // relu<<<re_grid, re_block>>>(f1_output, relu_output, 500);
        // float* test4 = new float[500];
        // cudaMemcpy(test4, relu_output, 500 * sizeof(float), cudaMemcpyDeviceToHost);
        // printArray(test4, 500);
        // //--------------------------------FC_2,1,1,500,10,1,1,0,0,5000,10,5010----------------------------------------------//
        // dim3 f2_block(10, 1, 1);
        // dim3 f2_grid(1, 1, 1); 
        // conv<<<f2_grid, f2_block>>>(f1_output, d_fc2, f2_output, 1, 1, 10);
        
        // float* output_softmax = new float[10];
        // cudaMemcpy(output_softmax, f2_output, 10 * sizeof(float), cudaMemcpyDeviceToHost);
        // float* output = new float[10]; 
        // printArray(output_softmax, 10);
        // printArray(output, 10);
        // softmax(output_softmax, output, 10);
        // printArray(output, 10);

        // delete[] output;
        // delete[] output_softmax;
        // delete[] input;

    
    

    cudaDeviceSynchronize();
    
    // conv<<<blocks, threads>>>(c_input, c_kernel, c_output, isize, ksize);

    // cudaMemcpy(outputMatrix, c_output, outputsize, cudaMemcpyDeviceToHost);
    
    cudaFree(c1_output);
    cudaFree(c2_output);
    cudaFree(p1_output);
    cudaFree(p2_output);
    
    
    // printMatrix(outputMatrix);

    return 0;
}

