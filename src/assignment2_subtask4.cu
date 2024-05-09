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
struct Record {
    uint32_t filenameLength;
    string filename;
};
vector<string> filenames_list;
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

//--------------------------------------DEVICE KERNEL CODE-----------------------------------------------------------//


//---------------------------------Convulation without Padding-----------------------------------------------------//
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
bool loadImagesFromBin(const std::string& filePath, float* imageArray, int numImages, int imageSize) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return false;
    }

    for (int i = 0; i < numImages; ++i) {
        uint32_t filenameLength;
        file.read(reinterpret_cast<char*>(&filenameLength), sizeof(filenameLength));
        if (file.fail()) {
            std::cerr << "Failed to read filename length." << std::endl;
            file.close();
            return false;
        }
        std::string filename(filenameLength, '\0');
        file.read(&filename[0], filenameLength);
        if (file.fail()) {
            std::cerr << "Failed to read filename." << std::endl;
            file.close();
            return false;
        }
        // Remove .png extension
        size_t pos = filename.find(".png");
        if (pos != std::string::npos) {
            filename.erase(pos, 4);
        }
        filenames_list.push_back(filename);
        file.read(reinterpret_cast<char*>(imageArray + i * imageSize * imageSize), imageSize * imageSize * sizeof(float));
        if (file.fail()) {
            std::cerr << "Failed to read image data for " << filename << "." << std::endl;
            file.close();
            return false;
        }
    }

    file.close();
    return true;
}
//---------------------------------------IMAGE PROCESSING CODE----------------------------------------------------------//
void image_processing_batch(float* c1_input, float* c1_output, float* p1_output, float* c2_output, float* p2_output, float* f1_output, float* f2_output, float* d_conv1, float* d_conv2, float* d_fc1, float* d_fc2, int batch_size, int isize,vector<cudaStream_t> &streams, int stream_idx, int i){
    //---------LENET architecure with Layer,Input_dim,output_dim,Input_Channels,Output_Channels,Kernel,Stride,Padding,Has Relu ?,No of Weights,Bias,Total Weights-------------------------------------------//                      
    //---------------------------------------------------------------------------------------------------------------------//
    for (int imgIdx = 0; imgIdx < batch_size; ++imgIdx) {
        // Calculate pointers to the current image in the input and output buffers
        float* curr_c1_input = c1_input + imgIdx * isize;
        float* curr_c1_output = c1_output + imgIdx * 24*24*20;
        float* curr_p1_output = p1_output + imgIdx * 12*12*20;
        float* curr_c2_output = c2_output + imgIdx * 8*8*50;
        float* curr_p2_output = p2_output + imgIdx * 4*4*50;
        float* curr_f1_output = f1_output + imgIdx * 500;
        float* curr_f2_output = f2_output + imgIdx * 10;

        //cout << "new f1 c1 c2 p1 p2 are allocated" <<  endl;

    //--------------------------------conv1-28,24,1,20,5,1,0,0,500,20,520--------------------------------------------------------//
        int out_channel = 20;
        dim3 c1_block(16, 16, 1);
        dim3 c1_grid((24 + c1_block.x - 1) / c1_block.x,
                (24 + c1_block.y - 1) / c1_block.y,
                out_channel);
        conv_cuda<<<c1_grid, c1_block, 0, streams[stream_idx]>>>(curr_c1_input, d_conv1, curr_c1_output, 1, 20, 28, 5);
            // Wait for the kernel to complete
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "Error during conv_cuda_1 execution: " << cudaGetErrorString(error) << std::endl;
        }
        //cout << "conv1_done" << endl;
        // //-----------------------------------Pool_1,24,12,20,20,2,2,0,0,-,-,--------------------------------------------------------------//
        dim3 p1_block(16, 16, 1); 
        dim3 p1_grid((12 + p1_block.x - 1) / p1_block.x,
                (12 + p1_block.y - 1) / p1_block.y,
                out_channel);
        MaxPooling<<<p1_grid, p1_block, 0, streams[stream_idx]>>>(curr_c1_output, curr_p1_output, 20, 24, 2, 2);      
        cudaError_t error1 = cudaGetLastError();
        if (error1 != cudaSuccess) {
            std::cerr << "Error during Max_pool_1 execution: " << cudaGetErrorString(error1) << std::endl;
        }
        //cout << "pool1 done " << endl;
        // //------------------------------------Conv_2,12,8,20,50,5,1,0,0,25000,50,25050--------------------------------------------------------//
        out_channel = 50;
        dim3 c2_block(8, 8); // One thread per output pixel
        dim3 c2_grid((8 + c2_block.x - 1) / c2_block.x,
                (8 + c2_block.y - 1) / c2_block.y,
                out_channel); // 50 channels to process && batch_size/cuda_stream imgs

        conv_cuda<<<c2_grid, c2_block, 0, streams[stream_idx]>>>(curr_p1_output, d_conv2, curr_c2_output, 20, 50, 12, 5);
        
        cudaError_t error2 = cudaGetLastError();
        if (error2 != cudaSuccess) {
            std::cerr << "Error during conv_cuda_2 execution: " << cudaGetErrorString(error2) << std::endl;
        }
        //cout << "conv2 done" << endl;

        // //------------------------------------------Pool_2,8,4,50,50,2,2,0,0,-,-,-----------------------------------------------//
            
        dim3 p2_block(4, 4);
        dim3 p2_grid(1, 1, out_channel);
        MaxPooling<<<p2_grid, p2_block, 0,streams[stream_idx]>>>(curr_c2_output, curr_p2_output, 50, 8, 2, 2);
        
        cudaError_t error3 = cudaGetLastError();
        if (error3 != cudaSuccess) {
            std::cerr << "Error during Max_pool_2 execution: " << cudaGetErrorString(error3) << std::endl;
        }
        //cout << "pool 2 done" << endl;
        // //--------------------------------------------FC_1,4,1,50,500,4,1,0,1,400000,500,400500-------------------------------------//
            
        int f1_block = 256;
        int f1_grid = (500 + f1_block - 1) / f1_block;
        dim3 threads1(f1_block);
        dim3 blocks1(f1_grid);
        fconv_cudaR<<<blocks1, threads1, 0, streams[stream_idx]>>>(curr_p2_output, d_fc1, curr_f1_output, 50, 500, 4);
        cudaError_t error4 = cudaGetLastError();
        if (error4 != cudaSuccess) {
            std::cerr << "Error during fully_conv_cuda_1 execution: " << cudaGetErrorString(error4) << std::endl;
        }
        //cout <<  "fc1 done" << endl;

        // //----------------------------------------FC_2,1,1,500,10,1,1,0,0,5000,10,5010-----------------------------------------------------//
            // printArray(test6, 10);
        
        dim3 threads(256);
        dim3 blocks((10 + f1_block - 1) / f1_block);
        fconv_cuda<<<blocks, threads, 0, streams[stream_idx]>>>(curr_f1_output, d_fc2, curr_f2_output, 500, 10, 1);
        cudaError_t error5 = cudaGetLastError();
        if (error5 != cudaSuccess) {
            std::cerr << "Error during fully_conv_cuda_2 execution: " << cudaGetErrorString(error5) << std::endl;
        }
            
            //cout << "fc2  done" << endl;
        // //------------------------------------SOFTMAX----------------------------------------------------//

        float* test6 = new float[10];
        CHECK_CUDA_ERROR(cudaMemcpy(test6, curr_f2_output, 10 * sizeof(float), cudaMemcpyDeviceToHost));
        //printArray(test6, 10 );
        softmax(test6, 10);
        //printArray(test6,10 );
    
        vector<int> top_classes;
        vector<float> top_probs;
        findTop5(test6, 10, top_classes, top_probs);
        //cout << count << endl;
        
        std::string outputFilename = "output/" + filenames_list[imgIdx+(i*100)] + "_softmax.txt";
        //cout << outputFilename << endl;
        std::ofstream outFile(outputFilename);

        if (!outFile.is_open()) {
            std::cerr << "Failed to open the file for writing: " << outputFilename << std::endl;
            continue;
        }
        for (size_t i = 0; i < top_classes.size(); ++i) {
            outFile << top_probs[i] << " class " << top_classes[i] << std::endl;
        }
        outFile.close();
            // //-----------------------------------Delete Memeory-----------------------------------------------------//   
    }                        
}
  

//------------------------------------MAIN FUNCTION-------------------------------------------------//
int main(int argc, char* argv[]) {
    //int correct_output = 0;
    
    const int num_streams = 16;
    std::vector<cudaStream_t> streams(num_streams);
     
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    std::vector<std::string> filepaths;
    const int batch_size = 100;
    const int isize = 28 * 28;
    string saveDirectory = "output/";
    string fileExtension = ".txt";
    //-----------------------Reading Trained Weights in Weights struct datatype ----------------------//
    Weights weights;
    weights.conv1 = fileReadP("weights/conv1.txt", 520);
    weights.conv2 = fileReadP("weights/conv2.txt", 25050);
    weights.fc1 = fileReadP("weights/fc1.txt", 400500);
    weights.fc2 = fileReadP("weights/fc2.txt", 5010);
    //printArray(weights.conv1, 520);
    float *d_conv1, *d_fc2, *d_conv2, *d_fc1; //variable for Device holding pointer to the data of conv1.txt, conv2.txt fc1.txt and fc2.txt respectively
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv1, 520 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv1, weights.conv1, 520 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc(&d_conv2, 25050 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_conv2, weights.conv2, 25050 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc1, 400500 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc1, weights.fc1, 400500 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fc2, 5010 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_fc2, weights.fc2, 5010 * sizeof(float), cudaMemcpyHostToDevice));
             
    
    std::string directory = "pre-proc-img/";
    DIR* dir;
    struct dirent* ent;
    cudaEvent_t start, stop;
    int count = 0;
    int ar = stoi(argv[1]);
    //int out_channel = 20;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

     
    
    if ((dir = opendir(directory.c_str())) != nullptr) {    
   
        while ((ent = readdir(dir)) != nullptr) {
            count++;
            
            std::string filename = ent->d_name;
            if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".bin") {
                std::string filepath = directory + filename;
                //cout << filepath  << endl;
                string labelpath = "/home/cse/btech/cs1190378/MNIST-CNN-LeNet-CUDA/pre-proc-img/labels_batch_binary/" + filename.substr(0, 7) + "_labels.txt";
                filepaths.push_back(filepath);
                //cout << filepath  << endl;        
            }       
            
        }
        closedir(dir);    
    }         
    else {
        // Could not open directory
        std::cerr << "Could not open directory" << std::endl;
        return EXIT_FAILURE;
    }     
    int stream_idx = 0; // Index of the current stream to use
    for (int i = 0 ; i < 100 ; i++) {
        float *c1_output, *p1_output, *c2_output, *p2_output, *f1_output, *f2_output, *c1_input;
        CHECK_CUDA_ERROR(cudaMalloc(&c1_input, 28*28*batch_size * sizeof(float)));  
        CHECK_CUDA_ERROR(cudaMalloc(&c1_output, 24*24*20*100*sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&p1_output, 12*12*20*100*sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&c2_output, 8*8*50*100*sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&p2_output, 4*4*50*100*sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&f1_output, 500*100*sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&f2_output, 10 * 100*sizeof(float)));

        float* input = new float[28*28*100];
        //cout << filepaths[i] << endl;
        loadImagesFromBin(filepaths[i], input, 100, 28);
        //printArray(input, 28*28);
        CHECK_CUDA_ERROR(cudaMemcpy(c1_input, input,28*28*100*sizeof(float), cudaMemcpyHostToDevice));


        //cout << filepaths[i] << endl;
        // std::string savePath = saveDirectory + filepaths[i].substr(filepaths[i].length() - 12, 8) + "_top5" + fileExtension;
        // //cout  <<  savePath << endl;
        // std::ofstream outFile(savePath, std::ios::out); // Open in write mode, overwrites existing file
        // if (!outFile.is_open()) {
        //     std::cerr << "Failed to open the file for writing." << std::endl;
        //     return -1; // Handle error appropriately
        // }



        image_processing_batch(c1_input, c1_output, p1_output, c2_output, p2_output, f1_output, f2_output, d_conv1, d_conv2, d_fc1, d_fc2, batch_size, isize, streams, stream_idx, i);
        if(ar==0){
            stream_idx = 0;
        }
        else if (ar==1){
            stream_idx = (stream_idx + 1) % num_streams;
        }
        
        //cout<< stream_idx << endl;

        //outFile.close();  
        CHECK_CUDA_ERROR(cudaFree(c1_output));
        CHECK_CUDA_ERROR(cudaFree(c2_output));
        CHECK_CUDA_ERROR(cudaFree(p1_output));
        CHECK_CUDA_ERROR(cudaFree(p2_output));    
        CHECK_CUDA_ERROR(cudaFree(f1_output));
        CHECK_CUDA_ERROR(cudaFree(f2_output));

    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

   //cout << count << endl;
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

