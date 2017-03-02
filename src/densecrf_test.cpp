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
#define DCNEG_FASTAPPROX false

void DenseCRF::greedyAlgorithmTest(MatrixXf & Qs,  MatrixXf & Q){
	
	MatrixXf negGrad( M_, N_ );
	getNegGradient(negGrad, Q); //negative gradient

	Qs = MatrixXf::Zero(Qs.rows(), Qs.cols());	

	MatrixXf temp(M_, N_), rescaled_Q(M_, N_);
	MatrixXf prod_dc = MatrixXf::Constant(M_, N_, 0);
	MatrixXf prod_filter = MatrixXf::Constant(M_, N_, 0);

	Map<VectorXf> w(negGrad.data(), negGrad.size());

	int len = negGrad.size();
	std::vector<float> wSorted;

	for(int i = 0; i < len; i++)
		wSorted.push_back(w[i]);

	sort(wSorted.begin(), wSorted.begin() + len);
	reverse(wSorted.begin(), wSorted.begin() + len);

	std::vector<int> indexVec(len);
	size_t n(0);
        generate(begin(indexVec), end(indexVec), [&]{ return n++; });
        sort(begin(indexVec), end(indexVec), [&](int i1, int i2) { return w[i1] > w[i2]; } );

	std::vector<std::vector<int> > sigma;

	for(int j = 0; j < M_; j++){
		std::vector<int> sigmaj;
		int countj = N_;
		for(int k = 0; k < len; k++){
			int index = indexVec[k];
			if(index % M_ == j){
				int i = (int) index/M_;		
				temp(j, i) = countj; 
				countj = countj - 1;
				sigmaj.push_back(index); //indexing from 0 - (NL -1)
			}
		}
		sigma.push_back(sigmaj);
	}

	rescale(rescaled_Q, temp);
        sortRows(Q, ind);

	for( unsigned int k=0; k<pairwise_.size(); k++ ){
            pairwise_[k]->apply_upper_minus_lower_ord(prod_filter, rescaled_Q);
		pairwise_[k]->apply_upper_minus_lower_dc(prod_dc, ind);
	std::cout << "Difference norm = " << (prod_filter - prod_dc).norm() << std::endl;
        }
}
