#include <vector>
#include <iostream>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;

void printMatrix(const vector<vector<float> >& matrix) {
    for (const auto& row : matrix) {
        for (const auto& elem : row) {
            cout << elem << " ";
        }
        cout << endl;
    }
}
vector<vector<float> > conv(vector<vector<float> > input, vector<vector<float> > kernel){
    int isize = input.size();
    int ksize = kernel.size();
    int res = isize-ksize+1;

    vector<vector<float> > outputMatrix(res, vector<float>(res, 0));
    for (int i = 0; i < res; ++i) {
        for (int j = 0; j < res; ++j) {
            float sum = 0;
            for (int m = 0; m < ksize; ++m) {
                for (int n = 0; n < ksize; ++n) {
                    sum += input[i + m][j + n] * kernel[m][n];
                }
            }
            outputMatrix[i][j] = sum;
        }
    }

    return outputMatrix;
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
    printMatrix(padMatrix);

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

void ReLU(vector<vector<float> >& a) {
    for (auto& row : a) {
        for (auto& i : row) {
            i = max(0.0f, i);
        }
    }
}

void Tanh(vector<vector<float> >& a) {
    for (auto& row : a) {
        for (auto& i : row) {
            i = tanh(i);
        }
    }
}

vector<vector<float> > MaxPol(const vector<vector<float> > input, int poolsize){
    int isize = input.size();
    int res= isize/poolsize;

    vector<vector<float> > outputMatrix(res, vector<float>(res, 0));

    for(int i =0; i<res; i++){
        for(int j =0; j<res;j++){
            float sum =0;
            for(int m =0; m<poolsize; m++){
                for(int n=0; n<poolsize; n++){
                    sum+=input[i*poolsize+m][j*poolsize+n];
                }
            }
            outputMatrix[i][j]=sum/(poolsize*poolsize);
        }
        
    }
    return outputMatrix;
}

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
vector<vector<float> > fileread(ifstream& file) {
    int rows, cols;
    file >> rows >> cols; 
    vector<vector<float> > matrix(rows, vector<float>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix[i][j];
        }
    }
    return matrix;
}

int main() {
    ifstream file("matrix.txt");
    if (!file.is_open()) {
        std::cerr << "Error opening file" << std::endl;
        return -1;
    }
    
    vector<vector<float> > input = fileread(file);
    vector<vector<float> > kernel = fileread(file);
    printMatrix(input);
    printMatrix(kernel);
    vector<vector<float> > output = MaxPol(input, 3);
    cout << "Output Matrix:" << endl;
    printMatrix(output);

    return 0;
}

