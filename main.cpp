#include <vector>
#include <iostream>

using namespace std;


vector<vector<float>> conv(vector<vector<float>> input, vector<vector<float>> kernel){
    int isize = input.size();
    int ksize = kernel.size();
    int res = isize-ksize+1;

    vector<vector<float>> outputMatrix(res, vector<float>(res, 0));
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

vector<vector<float>> convpad(vector<vector<float>> input, vector<vector<float>> kernel){
    int isize = input.size();
    int ksize = kernel.size();
    int res = isize-ksize+1;
    int pad = (ksize-1)/2;
    int padinput = isize + pad*2;
    int res = isize;

    vector<vector<float>> padMatrix(padinput, vector<float>(padinput, 0));

    vector<vector<float>> outputMatrix(res, vector<float>(res, 0));

    for (int i = 0; i < isize; ++i) {
        for (int j = 0; j < isize; ++j) {
            padMatrix[i + pad][j + pad] = input[i][j];
        }
    }

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

void ReLU(vector<vector<float>>& a) {
    for (auto& row : a) {
        for (auto& i : row) {
            i = max(0.0f, i);
        }
    }
}

void Tanh(vector<vector<float>>& a) {
    for (auto& row : a) {
        for (auto& i : row) {
            i = tanh(i);
        }
    }
}