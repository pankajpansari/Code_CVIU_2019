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

 
int main(int argc, char *argv[]){

    //process command-line arguments
    assert(argc == 7 && "example usage: ./inference_sparse_synthetic unary_0.txt 1 100 20 /home/pankaj/SubmodularInference/data/working/02_10_2017 1");

    std::string arg1(argv[1]);
    float weight = std::stof(argv[2]);

    //fixed 100x100 grid and 20 labels for synthetic experiment
    int H = std::stoi(argv[3]);
    int M = std::stoi(argv[4]);

    std::string log_dir(argv[5]);
    
    int good = std::stoi(argv[6]);
    
    //create sparse crf
    SparseCRF crf(H, H, M);

    //set unary
    MatrixXf unary = MatrixXf::Zero(M, H*H);
    std::string unary_file = "/home/pankaj/SubmodularInference/data/input/unaries/" + arg1;
    load_unary_synthetic(unary_file, H*H, M, unary);
    crf.setUnary(unary);

    //set pairwise
    crf.setPottsWeight(weight);
    
    //do inference
    std::cout << "unary: " << argv[1] << " weight: " << std::to_string(float(weight)) << std::endl; 
    std::string log_file = log_dir + "/submodular_log_w_" + std::to_string(float(weight)) + "_" + arg1;
        log_file = log_dir + "/submodular_log_w_" + std::to_string(float(weight)) + "_" + std::to_string(good) + "_" + arg1;

    crf.submodularFrankWolfe_Potts(unary, H, log_file, good);


}
