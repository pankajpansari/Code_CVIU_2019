#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <ctime>
#include "densecrf.h"
#include <math.h> 
#include "densecrf_utils.h"
#include "permutohedral.h"
#include "pairwise.h"
#include "file_storage.hpp"
#include <assert.h>

void DenseCRF::getConditionalGradient_rhst(MatrixXf &Qs, MatrixXf & Q, const std::string filename, const MatrixXf &unary_meta){

    using namespace std;
    M_ = Q.rows(); 
    int nMeta = Q.rows();
    int nvar = Q.cols();

    //get unaries
    MatrixXf negGrad = MatrixXf::Zero( nMeta, nvar);
    getNegGradient_rhst(negGrad, Q, filename); //negative gradient
    //get pairwise
    MatrixXf pairwise = MatrixXf::Zero(nMeta, nvar);
    applyFilter_rhst(pairwise, negGrad);
   
    //applyBruteForce(pairwise, negGrad);

    int nLabel, temp;
    float node_weight;
    ifstream treefile(filename);
    string s;

    getline(treefile, s);
    istringstream ss(s);
    ss >> temp >> nLabel;
    for(int i = 0; i < nMeta + nLabel; i++){
        getline(treefile, s);
    }
    for(int i = 0; i < nMeta; i++){
        getline(treefile, s);
        istringstream ss(s);
        ss >> temp >> node_weight;
        pairwise.row(temp) = pairwise.row(temp).array() * node_weight;
    }
    Qs = unary_meta - pairwise;
}

MatrixXf DenseCRF::submodular_inference_rhst( MatrixXf & init, int width, int height, std::string output_path, std::string tree_file, std::string dataset_name){

    std::cout << "submodular inference starting" << std::endl;
    std::string filename = tree_file;
    std::ifstream treefile(filename);
    std::string s;

    std::getline(treefile, s);
    std::istringstream ss(s);

    int nMeta, nLabel;
    ss >> nMeta >> nLabel;

    std::cout << "nMeta = " << nMeta << "       nLabel = " << nLabel << "       N_ = " << N_ << std::endl;

    MatrixXf Lmarg = MatrixXf::Zero(nMeta, N_); //Lmarg for log marginals
    MatrixXf Lmargs = MatrixXf::Zero(nMeta, N_);
    MatrixXf temp = MatrixXf::Zero(nMeta, N_);
    MatrixXf Lmargs_bf = MatrixXf::Zero(nMeta, N_);
    MatrixXf negGrad = MatrixXf::Zero( nMeta, N_ );
    MatrixXf unary_meta = MatrixXf::Zero( nMeta, N_ );
    
    MatrixXf marginal(nLabel, N_);
    MatrixXf labelLmarg(nLabel, N_);
    //log file
    std::string log_output = output_path;
    log_output.replace(log_output.end()-4, log_output.end(),"_log.txt");
    std::ofstream logFile;
    logFile.open(log_output);
    //
    //image file
    std::string image_output = output_path;
    std::string Q_output = output_path;

    img_size size;
    size.width = width;
    size.height = height;

    if(init.rows() != nLabel)
        std::cerr << "#rows of unary and nLabel different" << std::endl; 
    assert(init.rows() == nLabel); 
    assert(1 == 2); 
    assert(init.cols() == N_);
    
    unary_meta.block(0, 0, nLabel, N_) = init.block(0, 0, nLabel, N_);	 //augment unaries to have meta-labels as well (0 potentials)
    Lmarg = unary_meta;     //initialize to unaries
    float step = 0.001;

    clock_t start;
    float duration = 0;
    float objVal = 0;
    start = clock();

    objVal = getObj_rhst(Lmarg, filename);
    std::cout << "Iter: 0   Obj value = " << objVal << "  Step size = 0    Time = 0s" << std::endl;

    logFile << "0 " << " " << objVal << " " <<  duration << " " << step << std::endl;
    for(int k = 1; k <= 100; k++){

      getConditionalGradient_rhst(Lmargs, Lmarg, filename, unary_meta);

      step = doLineSearch_rhst(Lmargs, Lmarg, k, step, filename);

      Lmarg = Lmarg + step*(Lmargs - Lmarg); 

      duration = (clock() - start ) / (double) CLOCKS_PER_SEC;

      objVal = getObj_rhst(Lmarg, filename);

      if(k % 10 == 0){
                //name the segmented image and Q files
                std::string img_file_extn = "_" + std::to_string(k) + ".png";
                image_output = output_path;
                image_output.replace(image_output.end()-4, image_output.end(), img_file_extn);
                //save segmentation
                getMarginals_rhst(marginal, Lmarg, filename); 
                save_map(marginal, size, image_output, dataset_name);

      }
      std::cout << "Iter: " << k << " Obj = " << objVal << " Time = " <<  duration << " Step size = " << step << std::endl;
      logFile << k << " " << objVal << " " <<  duration << " " << step << std::endl;
      }

   getMarginals_rhst(marginal, Lmarg, filename); 

   std::string marg_file = output_path;
   marg_file.replace(marg_file.end()-4, marg_file.end(), "_marginals.txt");
   save_matrix(marg_file, marginal, size);
   return marginal;

}
