#include <chrono>
#include <fstream>
#include <string>
#include <cstddef>
#include <vector>
#include "densecrf.h"
#include "sparsecrf.h"
#include "file_storage.hpp"
#undef NDEBUG
#include <assert.h>

int main(){
    int N = 4; 
    int M = 2;

    SparseCRF crf(N, N, M);
    MatrixXf unary = crf.getUnary();
    crf.setPottsWeight(1);
    crf.readUnary("/home/pankaj/SubmodularInference/data/working/11_09_2017/unary.txt", M, N);

//    std::cout << crf.getUnary() << std::endl; 
//    MatrixXf unary = crf.readUnary("/home/pankaj/SubmodularInference/data/working/11_09_2017/unary.txt", M, N);
    int grid_size = N; 
    crf.submodularFrankWolfe(unary, grid_size);
}
