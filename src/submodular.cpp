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

void DenseCRF::greedyAlgorithm(MatrixXf &out, MatrixXf &grad){

    //negative gradient at current point is input
    //LP solution is the output

//    MatrixXf out1(M_, N_), out2(M_, N_);
//    std::cout << "Doing full filtering: "<< std::endl;
//    applyFullFilter(out1, grad);
//    std::cout << "Doing full bf: " << std::endl;
//    applyFullBruteForce(out2, grad);
//    MatrixP dot_tmp(M_, N_);
//    double costh = dotProduct(out1, out2, dot_tmp)/
//       (sqrt(dotProduct(out1, out1, dot_tmp))*sqrt(dotProduct(out2, out2, dot_tmp)));
//    std::cout << "Full bf-filter #cos-theta: " << costh << std::endl;

//    grad = grad.array() + 1;
//    grad = grad.array()/2;
//    assert(grad.minCoeff() >= 0);
//    assert(grad.maxCoeff() <= 1);

    out.fill(0);
    //get unaries
    MatrixXf unary = unary_->get();   
    
    //get pairwise
    MatrixXf pairwise = MatrixXf::Zero(M_, N_);
    
    clock_t start = std::clock();
    double duration = 0;
    applyFilter(pairwise, grad);

//   duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
//    std::cout<<"Filtering time: "<< duration <<'\n';


    out = unary + pairwise;

    //check equality constraint
//    float constraintDiff = out.sum() - unary.sum();
//    std::cout << "Pairwise sum = " << pairwise.sum() << std::endl;
//    std::cout << "out sum = " << out.sum() << std::endl;
//    std::cout << "unary sum = " << unary.sum() << std::endl;
//    std::cout << "Constraint difference = " << constraintDiff << std::endl; 
//
//    compareWithBf(pairwise, grad);
    //check singleton constraints
//    for(int j = 0; j < M_; j++)
//        for(int i = 0; i < N_; i++)
//            assert(out(j, i) <= getSubmodFnVal(j, i, unary) && "s(A) <= F(A) for singleton sets violated"); 
//    
    //project on plane if required
//    MatrixXf out_proj = out.array() - ((out - unary).sum()/(N_*M_));
//    out = out_proj; 
}

void DenseCRF::greedyAlgorithm_dc(MatrixXf &out, MatrixXf &grad){

    //negative gradient at current point is input
    //LP solution is the output

    out.fill(0);
    //get unaries
    MatrixXf unary = unary_->get();   
    
    //get pairwise
    MatrixXf pairwise = MatrixXf::Zero(M_, N_);
    applyFilter_dc(pairwise, grad);

    out = unary + pairwise;

}

void DenseCRF::greedyAlgorithm_bf(MatrixXf &out, MatrixXf &grad){

    //negative gradient at current point is input
    //LP solution is the output

    out.fill(0);
    //get unaries
    MatrixXf unary = unary_->get();   
    
    //get pairwise
    MatrixXf pairwise = MatrixXf::Zero(M_, N_);
//    grad = grad + MatrixXf::Constant(M_, N_, 0.5);
//    grad = grad.array()*PI;
//    grad = grad.tan()
    applyBruteForce(pairwise, grad);
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

    MatrixP dot_tmp(M_, N_);

    std::cout << "M_ = " << M_ << " N_ = " << N_ << std::endl;
    MatrixXf Qs_bf = MatrixXf::Zero(M_, N_);
    MatrixXf negGrad = MatrixXf::Zero( M_, N_ );

   Q = init;	//initialize to unaries

    float step = 0;
    img_size size;
    size.width = width;
    size.height = height;
//   PATH_TO_RESULT.replace(PATH_TO_RESULT.end()-4, PATH_TO_RESULT.end(),"_obj.txt");
//   PATH_TO_RESULT2.replace(PATH_TO_RESULT2.end()-4, PATH_TO_RESULT2.end(),"_time.txt");

    std::ofstream logFile;
    logFile.open(PATH_TO_RESULT + "/log.txt");

    clock_t start;
    float duration = 0;
    float scale = 0;
    float Qs_sum = 0;
    float dualGap = 0;
    float objVal = 0;
    start = clock();

    logFile << "0 " << getObj(Q) << " " <<  duration << " " << step << std::endl;
    std::cout << "Iter: " << 0 << "  Obj value = " << getObj(Q) << std::endl;

    for(int k = 1; k <= 100; k++){

      //for debugging purposes - getting max-marginal solutions

        std::cout << "Iter = " << k << std::endl;
      getNegGradient(negGrad, Q); //negative gradient

      greedyAlgorithm(Qs, negGrad);	
    
//      dualGap = dotProduct(-negGrad, Qs - Q, dot_tmp);

//      std::cout << "Dual gap = " << dualGap << std::endl;

      step = doLineSearch(Qs, Q, k);

      Q = Q + step*(Qs - Q); 

      if(k == 10 || k == 100){

            expAndNormalizeSubmod(temp, -Q);
            save_map(temp, size, PATH_TO_RESULT + "/segment" + std::to_string(k) + ".png", "MSRC");

            duration = (clock() - start ) / (double) CLOCKS_PER_SEC;

            objVal = getObj(Q);

            logFile << k << " " << objVal << " " <<  duration << " " << step << std::endl;

            std::cout << "Iter: " << (k) << "  Obj value = " << objVal << "  Step size = " << step << " Time = " << duration << std::endl;

            saveCurrentMarginals(Q, k);

      }

    }
   logFile.close();
   return Q;
}
