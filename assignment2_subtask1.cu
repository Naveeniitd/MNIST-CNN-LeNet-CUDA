#include <vector>
#include <iostream>
#include <numeric>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
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
//--------------------------Print Array and Matrix Function----------------------------------------//
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
//------------------------------------SOFTMAX Function---------------------------------------------------------//
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
//-------------------------------------SIGMOID Function-------------------------------------------------------------------//
void sigmoid(float* arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = 1.0f / (1.0f + exp(-arr[i]));
    }
}

//---------------------------------------Convolution without Padding--------------------------------------------------------//
void conv(const float* input, const float* weights, float* output, int in_channel, int out_channel, int isize, int ksize) { 
    int res = isize - ksize + 1;
    // Iterate over each output channel
    for(int o = 0; o < out_channel; ++o) {
        // Iterate over the output spatial dimension
        for(int y = 0; y < res; ++y) {
            for(int x = 0; x < res; ++x) {
                float sum = 0.0f; //accumulator for the sum        
                // Iterate over each input channel
                for(int c = 0; c < in_channel; ++c) {
                    // Convolve the kernel with the input
                    for(int ky = 0; ky < ksize; ++ky) {
                        for(int kx = 0; kx < ksize; ++kx) {
                            int iy = y + ky;
                            int ix = x + kx; // Calculate the input x-coordinate
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
//---------------------------------------Convolution with Padding--------------------------------------------------------//
void conv_pad(const float* input, const float* weights, float* output, int in_channel, int out_channel, int isize, int ksize, int padding) {
    
    // Calculate the size of the output with padding considered
    int res = isize - ksize + 2 * padding + 1;

    // Iterate over each output channel
    for(int o = 0; o < out_channel; ++o) {
        // Iterate over the output spatial dimension
        for(int y = 0; y < res; ++y) {
            for(int x = 0; x < res; ++x) {
                float sum = 0.0f; //accumulator for the sum 
                // Iterate over each input channel
                for(int c = 0; c < in_channel; ++c) {
                    // Convolve the kernel with the input (considering padding)
                    for(int ky = 0; ky < ksize; ++ky) {
                        for(int kx = 0; kx < ksize; ++kx) {
                            // Calculate the input coordinates considering the padding
                            int iy = y + ky - padding;
                            int ix = x + kx - padding;
                            // Check if the coordinates are within the bounds of the input dimensions (considering padding)
                            if (ix >= 0 && ix < isize && iy >= 0 && iy < isize) {
                                sum += input[c * isize * isize + iy * isize + ix] * weights[o * (in_channel * ksize * ksize) + c * (ksize * ksize) + ky * ksize + kx];
                            }
                            // If ix or iy are outside the bounds, they are effectively accessing the padding region,
                            // which is assumed to be zero, so we don't add anything to the sum.
                        }
                    }
                }
                output[o * res * res + y * res + x] = sum + weights[out_channel * in_channel * ksize * ksize + o];
            }
        }
    }
}

//------------------------------------_Fully connected Convolution---------------------------------------------------------//
void fconvR(const float* input, const float* weights, float* output, int in_channel, int out_channel, int isize) {
    // For FC_1, the output dimension is always 1 per filter
    int inputSize = isize * isize * in_channel; // Total size of the input for one filter
    
    for (int i = 0; i < out_channel; ++i) {
        const float* curr_weights = weights + i * inputSize; // Adjust index for multi-channel
        float bias = weights[out_channel * inputSize + i]; // Bias index adjusted for flattened input
        float sum = 0.0f;
        
        for (int j = 0; j < inputSize; ++j) {
            sum += input[j] * curr_weights[j];
        }
        
        // Apply bias and ReLU activation
        output[i] = max(0.0f, sum + bias);
    }
}
//------------------------------------_Fully connected Convolution without relu---------------------------------------------------------//
void fconv(const float* input, const float* weights, float* output, int in_channel, int out_channel, int isize) {
    // For FC_1, the output dimension is always 1 per filter
    int inputSize = isize * isize * in_channel; // Total size of the input for one filter
    
    for (int i = 0; i < out_channel; ++i) {
        const float* curr_weights = weights + i * inputSize; // Adjust index for multi-channel
        float bias = weights[out_channel * inputSize + i]; // Bias index adjusted for flattened input
        float sum = 0.0f;
        
        for (int j = 0; j < inputSize; ++j) {
            sum += input[j] * curr_weights[j];
        }
        
        // Apply bias 
        output[i] = sum + bias;
    }
}
//------------------------------------------Max Pooling-----------------------------------------------------------------------//
void MaxPooling(const float* input, float* output, int in_channel, int isize, int ksize, int stride) {
    // Calculate the dimension of the output feature maps
    int res = (isize - ksize) / stride + 1;

    // Iterate over each input channel
    for (int channel = 0; channel < in_channel; ++channel) {
        // Calculate the starting pointers for the current channel in the input and output
        const float* curr_input = input + channel * isize * isize;
        float* curr_output = output + channel * res * res;

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
                            maxVal = max(maxVal, curr_input[inputIndex]);
                        }
                    }
                }
                curr_output[y * res + x] = maxVal; // Assign the max value to the output
            }
        }
    }
}
//-----------------------------------------Average Pooling-----------------------------------------------------------------//
void AvgPooling(const float* input, float* output, int in_channel, int isize, int ksize, int stride) {
    // Calculate the dimension of the output feature maps
    int res = (isize - ksize) / stride + 1;
    // Iterate over each input channel
    for (int channel = 0; channel < in_channel; ++channel) {
        // Calculate the starting pointers for the current channel in the input and output
        const float* curr_input = input + channel * isize * isize;
        float* curr_output = output + channel * res * res;
        // Apply average pooling for the current channel
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                float sum = 0; // Initialize sum for calculating average
                int count = 0; // Counter for valid entries within the kernel window
                
                for (int ky = 0; ky < ksize; ++ky) {
                    for (int kx = 0; kx < ksize; ++kx) {
                        int iy = y * stride + ky; // Calculate the input's y-coordinate
                        int ix = x * stride + kx; // Calculate the input's x-coordinate
                        
                        // Check if the coordinates are within the bounds of the input size
                        if (ix < isize && iy < isize) {
                            int inputIndex = iy * isize + ix; // Calculate the index in the flattened input
                            sum += curr_input[inputIndex]; // Add the value to the sum
                            count++; // Increment the valid entry count
                        }
                    }
                }
                float avg = (count == 0) ? 0 : sum / count; // Calculate the average value
                curr_output[y * res + x] = avg; // Assign the average value to the output
            }
        }
    }
}
//-------------------------------------------Find Top 5 Probabilities-----------------------------------------------------------------//
void findTop5(const float* softmax_probs, int classes, std::vector<int>& topc, std::vector<float>& topprob) {
    // Using a min heap to store the top probabilities and their corresponding indices
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> min_heap;
    for (int i = 0; i < classes; ++i) {
        float prob = softmax_probs[i];
        if (min_heap.size() < 5) { // If the heap is not full
            min_heap.push({prob, i});
        } else if (prob > min_heap.top().first) { // If current prob is greater than the smallest in the heap
            min_heap.pop(); // Remove the smallest
            min_heap.push({prob, i}); // Insert the current
        }
    }
    // Extract the top 5
    topc.clear();
    topprob.clear();
    while (!min_heap.empty()) {
        topprob.push_back(min_heap.top().first);
        topc.push_back(min_heap.top().second);
        min_heap.pop();
    }
    // Reverse the order to make it from highest to lowest
    reverse(topc.begin(), topc.end());
    reverse(topprob.begin(), topprob.end());
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
//-----------------------------------Relu------------------------------------------------------//
void relu(float* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = max(0.0f, input[i]);
    }
}
//------------------------------------Tanh--------------------------------------------------------//
void tanh_activation(float* input, int size) {
    for (int i = 0; i < size; i++) {
        input[i] = tanh(input[i]);
    }
}
//------------------Function to load binary img in float* input----------------------------------//
bool imgload(const string& path, float* input){
    ifstream file(path.c_str(), ios::binary);
    file.read(reinterpret_cast<char*>(input), 28*28*sizeof(float));
    file.close();
    return true;
}
//--------------------------------_Main_Function_-------------------------------------------------//
int main(int argc, char* argv[]){
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [path_to_bin_file] [Image Size] [in_channel] [Function] [different parameter for different function]" << std::endl;
        return 1;
    }
    Weights weights;
    weights.conv1 = fileRead("weights/conv1.txt", 520);
    weights.conv2 = fileRead("weights/conv2.txt", 25050);
    weights.fc1 = fileRead("weights/fc1.txt", 400500);
    weights.fc2 = fileRead("weights/fc2.txt", 5010);
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
    file.read(reinterpret_cast<char*>(input), isize*isize*in_channel *sizeof(float));
    if (!file) {
        std::cerr << "Error reading file or file too short!" << std::endl;
        return 2;
    }

    file.close();


    if (strcmp(argv[4], "MaxPooling") == 0 ||strcmp(argv[4], "AvgPooling") == 0  ) {
        if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " [path_to_bin_file] [Image Size] [in_channel] [Function] [ksize] [stride]" << std::endl;
        return 1;
        }
        
        
        int ksize = stoi(argv[5]); 
        int stride = stoi(argv[6]);
        int res = (isize - ksize) / stride + 1;
        float* output = new float[res*res*in_channel];
        if(strcmp(argv[4], "MaxPooling") == 0 ){
        MaxPooling(input, output, in_channel, isize, ksize, stride);
        printArray(output,res*res*in_channel);}
        else{
            AvgPooling(input, output, in_channel, isize, ksize, stride);
            printArray(output,res*res*in_channel);
        }
    }
    else if (strcmp(argv[4], "Softmax") == 0 ||strcmp(argv[4], "Sigmoid") == 0  ) {
        if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " [path_to_bin_file] [Image Size] [in_channel] [Function]" << std::endl;
        return 1;
        }
        if(strcmp(argv[4], "Softmax") == 0 ){
        softmax(input, isize*isize*in_channel);
        printArray(input,isize*isize*in_channel);
        }
        else{
            sigmoid(input, in_channel*isize*isize);
            printArray(input, in_channel*isize*isize);
        }
    }
    else if (strcmp(argv[4], "relu") == 0 ||strcmp(argv[4], "tanh") == 0  ) {
        if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " [path_to_bin_file] [Image Size] [in_channel] [Function]" << std::endl;
        return 1;
        }
        if(strcmp(argv[4], "relu") == 0 ){
        relu(input, isize*isize*in_channel);
        printArray(input,isize*isize*in_channel);
        }
        else{
            tanh_activation(input, in_channel*isize*isize);
            printArray(input, in_channel*isize*isize);
        }
    }
    
    // //printArray(input, 28*28);
    // float* test_output = new float[24*24*20];
    // conv(input, weights.conv1, test_output, 1, 20, 28, 5);
    // //cout << weights.conv1.data() << " ";
    // printArray(test_output, 24*24*20 );
    // float* test2 = new float[12*12*20];
    // MaxPooling(test_output, test2 , 20, 24, 2, 2);
    // //printArray(test2,12*12*20 );
    // float* test3 = new float[8*8*50];
    // conv(test2,  weights.conv2, test3, 20, 50, 12, 5);
    // //printArray(test3, 8*8*50);
    // float* test4 = new float[4*4*50];
    // MaxPooling(test3, test4 , 50, 8, 2, 2);
    // //printArray(test4, 4*4*50);
    // float* test5 = new float[500];
    // fconvR(test4, weights.fc1, test5, 50, 500, 4);
    // //printArray(test5, 500);
    // float* test6 = new float[10];
    // fconv(test5, weights.fc2, test6, 500, 10, 1);
    // // printArray(test6, 10);
    // softmax(test6, 10);
    // printArray(test6, 10);
    // vector<int> topc;
    // vector<float> topprob;
    // findTop5(test6, 10, topc, topprob);
    // // Print top 5 probabilities and their classes
    // for (size_t i = 0; i < topc.size(); ++i) {
    //     std::cout << "Class " << topc[i] << " Probability: " << topprob[i] << std::endl;
    // }
    delete[] weights.conv1;
    delete[] weights.conv2;
    delete[] weights.fc1;
    delete[] weights.fc2;
    // delete[] test_output;
    // delete[] test2;
    // delete[] test3;
    // delete[] test4;
    // delete[] test5;
    // delete[] test6;
    delete[] input;

    return 0;

}
