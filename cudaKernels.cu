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
//------------------DEVICE CODE-------------------------------//
__global__ void conv(const float* input, const float* kernel, float* output, int isize, int ksize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
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



vector<vector<float> > convpad(vector<vector<float> > input, vector<vector<float> > kernel){
    int isize = input.size();
    int ksize = kernel.size();
    int pad = (ksize-1)/2;
    int padinput = isize + pad*2;
    int res = isize;

    vector<vector<float> > padMatrix(padinput, vector<float>(padinput, 0));

    vector<vector<float> > outputMatrix(res, vector<float>(res, 0));

    for (int i = 0; i < isize; ++i) {
        for (int j = 0; j < isize; ++j) {
            padMatrix[i + pad][j + pad] = input[i][j];
        }
    }
    //printMatrix(padMatrix);

    for (int i = 0; i < res; ++i) {
        for (int j = 0; j < res; ++j) {
            float sum = 0;
            for (int m = 0; m < ksize; ++m) {
                for (int n = 0; n < ksize; ++n) {
                    sum += padMatrix[i + m][j + n] * kernel[m][n];
                }
            }
            outputMatrix[i][j] = sum;
        }
    }

    return outputMatrix;
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
// vector<vector<float> > fileread(ifstream& file) {
//     int rows, cols;
//     file >> rows >> cols; 
    
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             file >> matrix[i][j];
//         }
//     }
//     return matrix;
// }
//--------------------------function to read trained weights---------------------------//
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
//------------------function to load binary img in float* input-----------------------//
bool imgload(const string& path, float* input){
    ifstream file(path.c_str(), ios::binary);
    file.read(reinterpret_cast<char*>(input), 28*28*sizeof(float));
    file.close();
    return true;
}

//----------------MAIN FUNCTION--------------------------//
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

    float* input = new float[28 * 28];
    cout << imgload("Binary_img/000000-num7.bin", input);

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

    
    float *c_input, *c_output;
    cudaMalloc(&c_input, 10000*28*28*sizeof(float));
    cudaMalloc(&c_output, 10000*28*28*sizeof(float));


    //cudaMemcpy(c_input, input.data(), 10000*28*28*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(c_kernel, flatkernel.data(), kernelsize, cudaMemcpyHostToDevice);

    // dim3 threads(16, 16);
    // dim3 blocks((res + threads.x - 1) / threads.x, 
    //                (res + threads.y - 1) / threads.y);

    // conv<<<blocks, threads>>>(c_input, c_kernel, c_output, isize, ksize);

    // cudaMemcpy(outputMatrix, c_output, outputsize, cudaMemcpyDeviceToHost);
    // cudaFree(c_input);
    // cudaFree(c_kernel);
    // cudaFree(c_output);

    // printMatrix(outputMatrix);

    return 0;
}

