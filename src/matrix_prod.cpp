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

#define VERBOSE false       // print intermediate energy values and timings, used in lp

MatrixXf DenseCRF::getFeatureMat(const unsigned char* im){

    MatrixXf feature( 5, N_ );
    for( int j=0; j<H_; j++ )
        for( int i=0; i<W_; i++ ){
            feature(0,j*W_+i) = i;
            feature(1,j*W_+i) = j;
            feature(2,j*W_+i) = im[(i+j*W_)*3+0];
            feature(3,j*W_+i) = im[(i+j*W_)*3+1];
            feature(4,j*W_+i) = im[(i+j*W_)*3+2];
       }
    featureMat_ = feature;
    return feature;
}

float DenseCRF::computeGaussianWeight(const VectorXf & feature_a, const VectorXf & feature_b){

    float spc_std = 1;
    float spc_potts = 7.467846;
    float bil_spcstd = 35.865959;
    float bil_colstd = 11.209644;
    float bil_potts = 4.028773;

    float dPos = pow((feature_a(0) - feature_b(0)), 2) + pow((feature_a(1) - feature_b(1)), 2);
    float dInt = pow((feature_a(2) - feature_b(2)), 2) + pow((feature_a(3) - feature_b(3)), 2)  + pow((feature_a(4) - feature_b(4)), 2);
    float bilVal = bil_potts*exp(-dPos/(2*pow(bil_spcstd, 2)) - dInt/(2*pow(bil_colstd, 2)));
    float spcVal = 	spc_potts*exp(-dPos/(2*pow(spc_std, 2)));
    return bilVal + spcVal;
}


void DenseCRF::applyBruteForce(MatrixXf &out, MatrixXf &Q){

	out.fill(0);
	int count = 0;
	for(int j = 0; j < M_; j++){
		for(int a = 0; a < N_; a++){
			VectorXf feature_a = featureMat_.col(a);
                        count = 0;
			for(int b = 0; b < N_; b++){
				if(a != b){ 
				VectorXf feature_b = featureMat_.col(b);
				if(Q(j, a) == Q(j, b))
					count = count + 1;
				if(Q(j, a) >= Q(j, b)){
					out(j, a) += computeGaussianWeight(feature_a, feature_b);}
				if (Q(j, a) <= Q(j, b)){
					out(j, a) -= computeGaussianWeight(feature_a, feature_b);	}
                                }
			}
		}	
	}	
}

void DenseCRF::applyFilter(MatrixXf &out, MatrixXf &in){
    //negative gradient at current point is input
    //matrix-matrix product is output

	out.fill(0);

        MatrixXf tmp(M_, N_), rescaled_in(M_, N_);

        rescale(rescaled_in, in);
    	for (int k = 0; k < pairwise_.size(); ++k) {
		pairwise_[k]->apply_upper_minus_lower_ord(tmp, rescaled_in);
		out += tmp;
	}
}

void DenseCRF::applyBruteForceAJ(MatrixXf &out, MatrixXf &Q){

	out.fill(0);

        MatrixXf tmp(M_, N_), tmp2(M_, N_);
	MatrixXi ind(M_, N_);

	sortRows(Q, ind);
    	for (int k = 0; k < pairwise_.size(); ++k) {
		no_norm_pairwise_[k]->apply_upper_minus_lower_bf_ord(tmp2, ind, Q);
		for(int i=0; i<tmp2.cols(); ++i) {
			for(int j=0; j<tmp2.rows(); ++j) {
				tmp(j, ind(j, i)) = tmp2(j, i);
			}
		}
		out += tmp;
	}
}


