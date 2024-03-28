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
struct Weights {
    vector<float> conv1;
    vector<float> conv2;
    vector<float> fc1;
    vector<float> fc2;
};

// void printMatrix(const vector<vector<float> >& matrix) {
//     for (const auto& row : matrix) {
//         for (const auto& elem : row) {
//             cout << elem << " ";
//         }
//         cout << endl;
//     }
// }
//------------------DEVICE KERNEL CODE-------------------------------//


//--------------convulation without padding---------------------------//
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

//-----------------------------Max Pooling--------------------------------------------//
__global__ void maxPool(const float* input, float* output, int isize, int output_channel, int poolSize, int stride) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < isize && y < isize && z < out_channel) {
        float maxVal = -FLT_MAX;
        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int X = x * stride + j;
                int Y = y * stride + i;
                if (X < isize && Y < isize) {
                    int index = z * isize * isize + Y * isize + X;
                    maxVal = fmaxf(maxVal, input[index]);
                }
            }
        }
        output[z * (isize / stride) * (isize / stride) + (y / stride) * (isize / stride) + (x / stride)] = maxVal;
    }
}


//---------HOST CODE-------------------------//
float sigmoid(float x){
    return 1.0f/(1.0f+exp(-x));
}

vector<float> sigfunc(const vector<float>& input){
    vector<float> output(input.size());
    transform(input.begin(), input.end(), output.begin(), sigmoid);
    return output;
}


vector<float> softmax(const vector<float>& input) {
    vector<float> outputMatrix(input.size());
    float p = *max_element(input.begin(), input.end());
    float sum = 0.0f;
    for (int i = 0; i < input.size(); ++i) {
        outputMatrix[i] = exp(input[i] - p); 
        sum += outputMatrix[i];
    }
    for (float& value : outputMatrix) {
        value /= sum;
    }

    return outputMatrix;
}


//--------------------------Function to read trained weights---------------------------//
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
    Weights weights;
    weights.conv1 = fileRead("trained_weights/conv1.txt");
    weights.conv2 = fileRead("trained_weights/conv2.txt");
    weights.fc1 = fileRead("trained_weights/fc1.txt");
    weights.fc2 = fileRead("trained_weights/fc2.txt");
    // cout << weights.conv1.size()<<" ";
    // cout << weights.conv2.size()<<" ";
    // cout << weights.fc1.size()<< " ";
    // cout << weights.fc2.size()<< " ";
    
    //-----------------------Host to Device Copy of Trained Weights---------------------------------------//

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
    float* input = new float[28 * 28];
    cout << imgload("Binary_img/000000-num7.bin", input);

    //-------------------Host to device copy of input/output---------------------------------------------------//
    float *c_input, *c_output;
    cudaMalloc(&c_input, 28*28*sizeof(float));
    cudaMalloc(&c_output, 28*28*sizeof(float));
    cudaMemcpy(c_input, input, 28*28*sizeof(float), cudaMemcpyHostToDevice);


    //---------LENET architecure with Layer,Input_dim,output_dim,Input_Channels,Output_Channels,Kernel,Stride,Padding,Has Relu ?,No of Weights,Bias,Total Weights-------------------------------------------//


    //----------------------------Conv_1 28,24,1,20,5,1,0,0,500,20,520-----------------------------//

    dim3 blockSize(16, 16, 1); // Block size of 16x16 for spatial dimensions and 1 for channels
    dim3 gridSize((24 + blockSize.x - 1) / blockSize.x, (24 + blockSize.y - 1) / blockSize.y, 20);

    conv<<<gridSize, blockSize>>>(c_input, d_conv1, c_output, 28, 5, 20);


    //----------------------------Pool_1,24,12,20,20,2,2,0,0,-,-,- -----------------------------------//
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((12 + blockSize.x - 1) / blockSize.x, (12 + blockSize.y - 1) / blockSize.y, 20);

    maxPool<<<gridSize, blockSize>>>(c_output, d_output, 24, 20, 2, 2);

    //----------------------------Conv_2,12,8,20,50,5,1,0,0,25000,50,25050-------------------------------------------//

    //----------------------------Pool_2,8,4,50,50,2,2,0,0,-,-,- ----------------------------------------------//
    dim3 blockSize(8, 8, 1);
    dim3 gridSize(1,1, 50);

    maxPool<<<gridSize, blockSize>>>(c_output, d_output, 24, 50, 2, 2);

    //------------------------------FC_1,4,1,50,500,4,1,0,1,400000,500,400500---------------------------------------//


    float* h_output = new float[24*24];
    cudaMemcpy(h_output, c_output, 24*24 * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 24; ++x) {
            std::cout << h_output[y * 24 + x] << " ";
        }
        std::cout << std::endl;
    }

    cudaDeviceSynchronize();
    // int isize = input.size();
    // int ksize = kernel.size();
    // int res = isize-ksize+1;
    
    
    // for(int i=0; i<isize; i++){
    //     for (int j=0; j<isize; j++){
    //         flatinput[isize*i+j] = input[i][j];
    //     }
    // }

    // for(int i=0; i<ksize; i++){
    //     for (int j=0; j<ksize; j++){
    //         flatkernel[ksize*i+j] = kernel[i][j];
    //     }
    // }

    // size_t inputsize = isize * isize * sizeof(float);
    // size_t kernelsize = ksize*ksize*sizeof(float);
    // size_t outputsize = res*res*sizeof(float);

    
    


    

    // dim3 threads(16, 16);
    // dim3 blocks((res + threads.x - 1) / threads.x, 
    //                (res + threads.y - 1) / threads.y);

    // conv<<<blocks, threads>>>(c_input, c_kernel, c_output, isize, ksize);

    // cudaMemcpy(outputMatrix, c_output, outputsize, cudaMemcpyDeviceToHost);
    cudaFree(c_input);
    cudaFree(c_output);

    delete[] h_output;

    // printMatrix(outputMatrix);

    return 0;
}

