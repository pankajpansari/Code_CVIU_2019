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

void DenseCRF::compareWithBf(MatrixXf &pairwise_filter, MatrixXf & grad){

    MatrixXf pairwise_bf = MatrixXf::Zero(M_, N_);

    std::cout << "Doing bf computation" << std::endl;
    applyBruteForce(pairwise_bf, grad);

   //check the angle
    MatrixP dot_tmp(M_, N_);
    double costh = dotProduct(pairwise_filter, pairwise_bf, dot_tmp)/
       (sqrt(dotProduct(pairwise_filter, pairwise_filter, dot_tmp))*sqrt(dotProduct(pairwise_bf, pairwise_bf, dot_tmp)));
    std::cout << "bf-filter #cos-theta: " << costh << std::endl;

    //check norms
    std::cout << "Filter o/p norm = " << pairwise_filter.norm() << std::endl;
    std::cout << "Bf o/p norm = " << pairwise_bf.norm() << std::endl;

}

void DenseCRF::getConditionalGradient(MatrixXf &Qs, MatrixXf & Q){
    //current solution is the input matrix (in)
    //conditional gradient is output

        M_ = Q.rows();
        N_ = Q.cols();

	MatrixXf negGrad( M_, N_ );
	getNegGradient(negGrad, Q); //negative gradient

        Qs.fill(0);
	greedyAlgorithm(Qs, negGrad);	
}

void DenseCRF::greedyAlgorithm(MatrixXf &out, MatrixXf &grad){

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
    pairwise = 0.5*pairwise.array();

    out = unary - pairwise; //-ve because original code makes use of negative Potts potential (in labelcompatibility.cpp), but we want to use positive weights

}


MatrixXf DenseCRF::submodular_inference( MatrixXf & init, int width, int height, std::string output_path){

    MatrixXf Q( M_, N_ ), Qs( M_, N_), temp(M_, N_); //Q is the current point, Qs is the conditional gradient

    MatrixP dot_tmp(M_, N_);

//    std::cout << "M_ = " << M_ << " N_ = " << N_ << std::endl;
    MatrixXf Qs_bf = MatrixXf::Zero(M_, N_);
    MatrixXf negGrad = MatrixXf::Zero( M_, N_ );

   Q = init;	//initialize to unaries

    float step = 0.001;
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
    std::string Q_output = output_path;

    clock_t start;
    float duration = 0;
    float scale = 0;
    float Qs_sum = 0;
    float dualGap = 0;
    float objVal = 0;
    start = clock();

    objVal = getObj(Q);
    logFile << "0 " << objVal << " " <<  duration << " " << step << std::endl;
  //  std::cout << "Iter: 0   Obj value = " << objVal << "  Step size = 0    Time = 0s" << std::endl;

    for(int k = 1; k <= 1000; k++){

      //for debugging purposes - getting max-marginal solutions

 //      std::cout << "Iter = " << k << std::endl;
        
    getConditionalGradient(Qs, Q);
//    getConditionalGradient_rhst(Qs, Q);
    
//      dualGap = dotProduct(-negGrad, Qs - Q, dot_tmp);

//      std::cout << "Dual gap = " << dualGap << std::endl;

      step = doLineSearch(Qs, Q, k, step);

      Q = Q + step*(Qs - Q); 

      duration = (clock() - start ) / (double) CLOCKS_PER_SEC;

      objVal = getObj(Q);

      //write to log file
      logFile << k << " " << objVal << " " <<  duration << " " << step << std::endl;
//      std::cout << k << " " << objVal << " " <<  duration << " " << step << std::endl;

      if(k % 20 == 0){
            //name the segmented image and Q files
            if(k == 10){
                image_output = output_path;
                Q_output = output_path;
                image_output.replace(image_output.end()-4, image_output.end(),"_10.png");
                Q_output.replace(Q_output.end()-4, Q_output.end(),"_Q_10.dat");
            }
            else{
                image_output = output_path;
                Q_output = output_path;
                image_output.replace(image_output.end()-3, image_output.end(),"png");
                Q_output.replace(Q_output.end()-4, Q_output.end(),"_Q.dat");
            }
 
          //save segmentation
           expAndNormalize(temp, -Q);
           save_map(temp, size, image_output, "Stereo_special");

            //write to console
 //           std::cout << "Iter: " << (k) << "  Obj value = " << objVal << "  Step size = " << step << " Time = " << duration << "s" << std::endl;
            
            //write the Q values
            write_binary(Q_output, Q);
      }
   }

   logFile.close();
   //convert Q to marginal probabilities
   MatrixXf marginal(M_, N_);
   expAndNormalize(marginal, -Q); 
 //  return Q;
   return marginal;
}
