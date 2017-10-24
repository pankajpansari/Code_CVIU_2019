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

void DenseCRF::getConditionalGradient_tree(MatrixXf &Qs, MatrixXf & Q, const std::vector<node> &G){

        M_ = Q.rows();
        N_ = Q.cols();

	MatrixXf negGrad( M_, N_ );
	getNegGradient_rhst(negGrad, Q, G); //negative gradient

        Qs.fill(0);
	greedyAlgorithm_tree(Qs, negGrad, G);	
}

void DenseCRF::greedyAlgorithm_tree(MatrixXf &out, MatrixXf &grad,  const std::vector<node> &G){

    //negative gradient at current point is input
    //LP solution is the output

    out.fill(0);
    //get unaries
    MatrixXf unary = unary_->get();   
    
    //get pairwise
    MatrixXf pairwise = MatrixXf::Zero(M_, N_);
    
    clock_t start = std::clock();
    double duration = 0;
    applyFilter(pairwise, grad);

    Eigen::VectorXf  weightVec = getWeight(G);

   for(int i = 0; i < pairwise.rows(); i++){
        pairwise.row(i) = pairwise.row(i).array() * weightVec[i];
   } 

    out = unary - pairwise; //-ve because original code makes use of negative Potts potential (in labelcompatibility.cpp), but we want to use positive weights
}

MatrixXf DenseCRF::submodularFrankWolfe_tree( MatrixXf & init, int width, int height, std::string output_path, std::string dataset_name, const std::vector<node> &G){
//M_ is number of meta-labels and L is number of labels

    MatrixXf Q = MatrixXf::Zero(M_, N_); //current point 
    MatrixXf temp = MatrixXf::Zero(M_, N_);
    MatrixXf Qs = MatrixXf::Zero(M_, N_);//conditional gradient

    MatrixP dot_tmp(M_, N_);

    node root = getRoot(G);
    std::vector<node> leaves = getLeafNodes(G);
    int L = leaves.size();
 
    Q.block(0, 0, L, N_) = init;

    float step = 0;
    img_size size;
    size.width = width;
    size.height = height;

    //log file
    std::string log_output = output_path;
    log_output.replace(log_output.end()-4, log_output.end(),"_log.txt");
    std::ofstream logFile;
    logFile.open(log_output);

    //image file
    std::string image_output = output_path;

    clock_t start;
    float duration = 0;
    float dualGap = 0;
    float objVal = 0;

    start = clock();

    objVal = getObj_rhst(Q, G);
    logFile << "0 " << objVal << " " <<  duration << " " << step << std::endl;

    for(int k = 1; k <= 100; k++){

      getConditionalGradient_tree(Qs, Q, G);

      step = doLineSearch_rhst(Qs, Q, 1, G);

      Q = Q + step*(Qs - Q); 

      duration = (clock() - start ) / (double) CLOCKS_PER_SEC;

     objVal = getObj_rhst(Q, G);

      //write to log file
      logFile << k << " " << objVal << " " <<  duration << " " << step << std::endl;
      std::cout << "Iter: " << k << " Obj = " << objVal << " Time = " <<  duration << " Step size = " << step << std::endl;
      
      if(k % 10 == 0){
            //name the segmented image and Q files
                std::string img_file_extn = "_" + std::to_string(k) + ".png";
                image_output = output_path;
                image_output.replace(image_output.end()-4, image_output.end(), img_file_extn);
                
             //save segmentation
               expAndNormalize_tree(temp, -Q, G);
               save_map(temp, size, image_output, dataset_name);

      }
   }

   logFile.close();

   //convert Q to marginal probabilities
   MatrixXf marginal(M_, N_);
   expAndNormalize_tree(marginal, -Q, G);

   return marginal;
}
