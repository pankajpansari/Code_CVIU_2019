#include <fstream>
#include <iostream>
#include "test.h"
#include <Eigen/Eigenvalues>
#include <cmath>

using namespace std;

void DenseCRF::greedyAlgorithm(MatrixXf & Qs, MatrixXf negGrad){
	
	//sort negative gradient and store the sorted indices
	//make all elements of Qs zero
	Qs = MatrixXf::Zero(Qs.rows(), Qs.cols());	

	//easier to work with vector format of -ve gradient
	MatrixXf unaryOrdered(M_, N_), unary(M_, N_), temp2(M_, N_), temp3(M_, N_);
	MatrixXf temp4 = MatrixXf::Constant(M_, N_, 0);
	Map<VectorXf> w(negGrad.data(), negGrad.size());

	int len = negGrad.size();
	std::vector<float> wSorted;

	//copy (to keep original w intact)
	for(int i = 0; i < len; i++){ 
		wSorted.push_back(w[i]);
	}

	//sort in descending order
	sort(wSorted.begin(), wSorted.begin() + len);
	reverse(wSorted.begin(), wSorted.begin() + len);

	std::vector<int> indexVec(len);
	size_t n(0);
        generate(begin(indexVec), end(indexVec), [&]{ return n++; });
        sort(begin(indexVec), end(indexVec), [&](int i1, int i2) { return w[i1] > w[i2]; } );


	std::vector<std::vector<int> > sigma(M_, std::vector<int> (N_));
	std::vector<std::vector<int> > nonZeroIndex(M_, std::vector<int> (N_));

	int i = 0, j = 0;
	for(int k = 0; k < len; k++){
		int index = indexVec[k];
		j = index % M_;
		i = (int) index/M_;		
		temp2(j, i) = index;
		sigma[j].push_back(i);
		nonZeroIndex[j].push_back(index);
	}

	unary = unary_->get();
	for(int j = 0; j < M_; j++){
		for(int i = 0; i < N_; i++){
			unaryOrdered(j, i) = unary(j, sigma[j][i]);
		}
	}

	Matrix<float ,Dynamic,Dynamic,RowMajor> Q_temp(temp2);
	for(int i = 0; i < Q_temp.rows(); i++)
		Q_temp.row(i) = Q_temp.row(i).array()/Q_temp.row(i).sum();
	

        temp3 = unaryOrdered;

	for( unsigned int k=0; k<pairwise_.size(); k++ ) {
            pairwise_[k]->apply_upper_minus_lower_ord( temp4, Q_temp );
            temp3 += temp4;
        }

	int currentNonZeroIndex = 0;
	for( unsigned int j=0; j<M_; j++ ) {
		for( unsigned int i=0; i<N_; i++ ) {
			currentNonZeroIndex  = nonZeroIndex[j][i];
			Qs(j,currentNonZeroIndex) = temp3(j, i);
		}
	}
}		
