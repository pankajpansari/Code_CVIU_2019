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

    //fixed 100x100 grid and 20 labels for synthetic experiment
    int H = 100; 
    int M = 20;

    //process command-line arguments
    assert(argc == 3 && "example usage: ./inference_sparse unary_0.txt 1");

    std::string arg1(argv[1]);

    float weight = std::stof(argv[2]);

    std::string unary_file = "/home/pankaj/Truncated_Max_of_Convex/data/input/synthetic/unaries/" + arg1;
    std::string log_file = "/home/pankaj/SubmodularInference/data/working/23_09_2017/synthetic_submodular_sparse/submodular_log_w_" + std::to_string(int(weight)) + "_" + arg1;
 
    //create sparse crf
    SparseCRF crf(H, H, M);

    //set unary
    MatrixXf unary = MatrixXf::Zero(M, H*H);
    load_unary_synthetic(unary_file, H*H, M, unary);
    crf.setUnary(unary);

    //set pairwise
    crf.setPottsWeight(weight);
    
    //do inference
    std::cout << "unary: " << argv[1] << " weight: " << std::to_string(int(weight)) << std::endl; 
    crf.submodularFrankWolfe(unary, H, log_file);
}
