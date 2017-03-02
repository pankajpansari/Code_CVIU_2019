#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include "densecrf.h"
#include <math.h> 
#include "densecrf_utils.h"
#include "permutohedral.h"
#include "pairwise.h"
#include "file_storage.hpp"

extern std::string PATH_TO_RESULT;

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

    //get unaries
    MatrixXf unary = unary_->get();   
    
    //get pairwise
    MatrixXf pairwise = MatrixXf::Zero(M_, N_);
//    grad = grad + MatrixXf::Constant(M_, N_, 0.5);
//    grad = grad.array()*PI;
//    grad = grad.tan()
    applyFilter(pairwise, grad);
//    std::cout << "Doing bf computation" << std::endl;
//    applyBruteForce(pairwise_bf, grad);
//
//    MatrixP dot_tmp(M_, N_);
//    double costh = dotProduct(pairwise, pairwise_bf, dot_tmp)/
//       (sqrt(dotProduct(pairwise, pairwise, dot_tmp))*sqrt(dotProduct(pairwise_bf, pairwise_bf, dot_tmp)));
//    std::cout << "bf-filter #cos-theta: " << costh << std::endl;
//    //check the angle

    out = unary + pairwise;
}

MatrixXf DenseCRF::submodular_inference( MatrixXf & init, int width, int height) {

    MatrixXf Q( M_, N_ ), Qs( M_, N_), temp(M_, N_); //Q is the current point, Qs is the conditional gradient

   Q = init;	//initialize to unaries

   MatrixXf unary = unary_->get();   
   const float unary_sum = unary.sum();

   std::string mapfile = "temp0.png";
   expAndNormalizeSubmod(temp, -Q);
   save_map(temp, {width, height}, mapfile, "MSRC");

    float step = 0;
//    PATH_TO_RESULT.replace(PATH_TO_RESULT.end()-4, PATH_TO_RESULT.end(),"_obj.txt");
 //   PATH_TO_RESULT2.replace(PATH_TO_RESULT2.end()-4, PATH_TO_RESULT2.end(),"_time.txt");

    std::ofstream logFile;
    logFile.open(PATH_TO_RESULT + "/log.txt");

    clock_t start;
    float duration = 0;
    float scale = 0;
    float Qs_sum = 0;
    start = clock();

    logFile << "0 " << getObj(Q) << " " <<  duration << " " << step << std::endl;
    std::cout << "Iter: " << 0 << "  Obj value = " << getObj(Q) << std::endl;

    for(int k = 1; k <= 10000; k++){

      //for debugging purposes - getting max-marginal solutions
      mapfile = PATH_TO_RESULT + "temp" + std::to_string(k) + ".png";
      expAndNormalizeSubmod(temp, -Q);
      save_map(temp, {width, height}, mapfile, "MSRC");

      getConditionalGradient(Qs, Q);		

//      scale = unary_sum/Qs.sum(); //for the equality constraint to be met
//      std::cout << "Scaling factor = " << scale << std::endl;

//      Qs = scale*Qs;

       step = doLineSearch(Qs, Q);

//	step = (float) 2/(k + 2);
        Q = Q + step*(Qs - Q); 

	duration = (clock() - start ) / (double) CLOCKS_PER_SEC;
        logFile << k << " " << getObj(Q) << " " <<  duration << " " << step << std::endl;
	std::cout << "Iter: " << (k) << "  Obj value = " << getObj(Q) << "  Step size = " << step << " Time = " << duration << std::endl;

    }
    logFile.close();
   return Q;
}
