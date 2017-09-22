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
    int H = 100; 
    int M = 20;

    SparseCRF crf(H, H, M);
    MatrixXf unary = MatrixXf::Zero(M, H*H);
    load_unary_synthetic("/home/pankaj/Truncated_Max_of_Convex/data/input/synthetic/unaries/unary_0.txt", H*H, M, unary);
 
    crf.setUnary(unary);
    crf.setPottsWeight(1);
    
//    std::cout << crf.getUnary() << std::endl; 
//    MatrixXf unary = crf.readUnary("/home/pankaj/SubmodularInference/data/working/11_09_2017/unary.txt", M, H);
//    int grid_size = H; 
    crf.submodularFrankWolfe(unary, H);
}
