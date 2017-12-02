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
#include "tree_utils.h"
#undef NDEBUG
#include <assert.h>

//const std::string FOLDER_NAME = "/home/pankaj/Truncated_Max_of_Convex/data/input/synthetic/unaries/"; 

int main(int argc, char *argv[]){

    //process command-line arguments
    assert(argc == 8 && "example usage: ./inference_sparse_synthetic unary_0.txt synthetic_tree_20_1.txt 1 100 20 /home/pankaj/SubmodularInference/data/working/03_10_2017 1");

    std::string arg1(argv[1]);
    std::string unary_file = "/home/pankaj/SubmodularInference/data/input/unaries/" + arg1;

    std::string arg2(argv[2]);
    std::string tree_file = "/home/pankaj/SubmodularInference/data/input/trees/short/" + arg2;

    float weight = std::stof(argv[3]); 

    int H = std::stoi(argv[4]); //grid size
    int L = std::stoi(argv[5]); //number of labels

    std::string log_dir(argv[6]); //log file location

    int good = std::stoi(argv[7]);

    //get number of meta-labels
    std::vector<node> G = readTree(tree_file);

    int M = getNumMetaLabels(G);
    int temp = getNumLabels(G, M); //checks for validity of leaf numbering as well (0 - (M -1))

    assert(temp == L && "Number of labels and leaves in tree do not agree");

    //create sparse crf
    SparseCRF crf(H, H, M, L);

    //set unary
    MatrixXf unary = MatrixXf::Zero(L, H*H);
    load_unary_synthetic(unary_file, H*H, L, unary);
    crf.setTreeUnary(unary);

    //set pairwise
    Eigen::VectorXf  weightVec = getWeight(G);
    crf.setTreeWeight(weight*weightVec);
    
    //do inference
    std::cout << "unary: " << argv[1] << " weight = " << weight << " rhst: " << arg2 << std::endl; 
    arg1.replace(arg1.end() - 4, arg1.end(), ""); //to remove .txt suffix
    std::string log_file = log_dir + "/submodular_log_w_" + std::to_string(float(weight)) + "_" + std::to_string(good) + "_" + arg1 + "_" + arg2;
    std::cout << "Log file = " << log_file << std::endl;
    crf.submodularFrankWolfe_tree(unary, H, log_file, G, good);
//    crf.submodularFrankWolfe(unary, H, log_file);

}
