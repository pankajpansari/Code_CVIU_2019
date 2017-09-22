#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <cstddef>
#include <vector>
#include <assert.h>
#include "densecrf.h"
#include "sparsecrf.h"
#include "file_storage.hpp"
#undef NDEBUG
#include <assert.h>

//const std::string FOLDER_NAME = "/home/pankaj/Truncated_Max_of_Convex/data/input/synthetic/unaries/"; 

int main(int argc, char *argv[]){
    int H = 100; 
    int M = 20;

    assert(argc == 3 && "usage: ./inference_sparse unary_0.txt 1");
    SparseCRF crf(H, H, M);
    MatrixXf unary = MatrixXf::Zero(M, H*H);

    std::string arg1(argv[1]);

    std::string unary_file = "/home/pankaj/Truncated_Max_of_Convex/data/input/synthetic/unaries/" + arg1;

    load_unary_synthetic(unary_file, H*H, M, unary);
 
    crf.setUnary(unary);
    
    float weight = std::stof(argv[2]);
    std::cout << weight << std::endl;
    crf.setPottsWeight(weight);
    
    crf.submodularFrankWolfe(unary, H);
}
