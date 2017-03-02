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

extern std::string PATH_TO_RESULT;
extern std::string PATH_TO_RESULT2;

void DenseCRF::applyBruteForce(MatrixXf &Qs_pairwise, MatrixXf &Q){

	Qs_pairwise.fill(0);

	MatrixXf negGrad(M_, N_);
	getNegGradient(negGrad, Q);

	int len = M_ * N_;

	Map<VectorXf> w(negGrad.data(), negGrad.size());

	std::vector<float> wSorted;

	//copy (to keep original w intact)
	for(int i = 0; i < len; i++)
		wSorted.push_back(w[i]);

	//sort in descending order
        std::sort(wSorted.begin(), wSorted.begin() + len);
        std::reverse(wSorted.begin(), wSorted.begin() + len);
       
	std::vector<int> indexVec(len);
	size_t n(0);
        std::generate(begin(indexVec), end(indexVec), [&]{ return n++; });
        std::sort(begin(indexVec), end(indexVec), [&](int i1, int i2) { return w[i1] > w[i2]; } );

	std::vector<std::vector<int> > sigma;

	for(int j = 0; j < M_; j++){
		std::vector<int> sigmaj;
		for(int k = 0; k < len; k++){
			int index = indexVec[k];
			if(index % M_ == j)
				sigmaj.push_back(index); //indexing from 0 - (NL -1)
		}
		sigma.push_back(sigmaj);
	}

	for(int j = 0; j < M_; j++)
	for(int a = 0; a < N_; a++){
		int var_a = (int) sigma[j][a]/M_;
		VectorXf feature_a = featureMat_.col(var_a);
		for(int b = 0; b < N_; b++){
			int var_b = (int) sigma[j][b]/M_;
			VectorXf feature_b = featureMat_.col(var_b);
			if(b > a)
				Qs_pairwise(j, a) += computeGaussianWeight(feature_a, feature_b)/2;
			else if (b < a)
				Qs_pairwise(j, a) += -computeGaussianWeight(feature_a, feature_b)/2;
			 else
				Qs_pairwise(j, a) += 0;
		}
	}
}

void DenseCRF::applyFilter(MatrixXf &Qs_pairwise, MatrixXf &Q){

	Qs_pairwise.fill(0);

	MatrixXf negGrad(M_, N_);
	getNegGradient(negGrad, Q);

	int len = M_ * N_;

	Map<VectorXf> w(negGrad.data(), negGrad.size());

	std::vector<float> wSorted;

	//copy (to keep original w intact)
	for(int i = 0; i < len; i++)
		wSorted.push_back(w[i]);

	//sort in descending order
        std::sort(wSorted.begin(), wSorted.begin() + len);
        std::reverse(wSorted.begin(), wSorted.begin() + len);
       
	std::vector<int> indexVec(len);
	size_t n(0);
        std::generate(begin(indexVec), end(indexVec), [&]{ return n++; });
        std::sort(begin(indexVec), end(indexVec), [&](int i1, int i2) { return w[i1] > w[i2]; } );

	MatrixXf temp = MatrixXf::Constant(M_, N_, 0);
	MatrixXf ind = MatrixXf::Constant(M_, N_, 0);
	MatrixXf ph_grad = MatrixXf::Constant(M_, N_, 0);
	MatrixXf bf_grad = MatrixXf::Constant(M_, N_, 0);
	MatrixXf rescaled_Q = MatrixXf::Constant(M_, N_, 0);

	for(int j = 0; j < M_; j++){
		int countj = N_;
		for(int k = 0; k < N_*M_; k++){
			int index = indexVec[k];
			if(index % M_ == j){
				int i = (int) index/M_;		
				ind(j, i) = countj; 
				countj = countj - 1;
			}
		}
	}
	
	compare_filter(ind, ph_grad, bf_grad); 
//	renormalize(ind);
//	rescale(rescaled_Q, ind);
//	for( unsigned int k=0; k<pairwise_.size(); k++ ) {
//	    pairwise_[k]->apply_upper_minus_lower_ord(temp, rescaled_Q);
//	    temp = temp/2;
//	    Qs_pairwise += temp;
//        }
}

void DenseCRF::compare_filter(MatrixXf & Q, MatrixXf & ph_grad, MatrixXf & bf_grad) const {
    renormalize(Q); // must renormalize before using it
	if (!valid_probability(Q)) {
		std::cout << "Q is not a valid probability!" << std::endl;
		exit(1);
	}
	
    MatrixXf tmp(M_, N_), tmp2(M_, N_), rescaled_Q(M_, N_);
	MatrixXi ind(M_, N_);
	MatrixP dot_tmp;
	double energy = 0;
    ph_grad.fill(0);
    bf_grad.fill(0);

    //ph-energy
    rescale(rescaled_Q, Q);
    for (int k = 0; k < pairwise_.size(); ++k) {
        // Add the upper minus the lower
        pairwise_[k]->apply_upper_minus_lower_ord(tmp, rescaled_Q);

        ph_grad -= tmp;
    }

	// bf-energy
	// for bruteforce computation --> use not normalized pairwise!
    // ""normalization" is something that is done when we initialize the kernel!
	sortRows(Q, ind);
    for (int k = 0; k < pairwise_.size(); ++k) {
        no_norm_pairwise_[k]->apply_upper_minus_lower_bf_ord(tmp2, ind, Q);
    	// need to sort before dot-product
    	for(int i=0; i<tmp2.cols(); ++i) {
        	for(int j=0; j<tmp2.rows(); ++j) {
            	tmp(j, ind(j, i)) = tmp2(j, i);
        	}
        }
        bf_grad -= tmp;
    }

    // should be coliner 
    MatrixXf ph_bf = ph_grad - bf_grad;
    double costh = dotProduct(ph_grad, bf_grad, dot_tmp)/
        (sqrt(dotProduct(ph_grad, ph_grad, dot_tmp))*sqrt(dotProduct(bf_grad, bf_grad, dot_tmp)));
    std::cout << "#cos-theta: " << costh << std::endl;
    std::cout << "BF   :: mean=" << bf_grad.mean() << ",\tmax=" << bf_grad.maxCoeff() << ",\tmin=" << bf_grad.minCoeff() << std::endl;
    std::cout << "PH   :: mean=" << ph_grad.mean() << ",\tmax=" << ph_grad.maxCoeff() << ",\tmin=" << ph_grad.minCoeff() << std::endl;
    std::cout << "PH-BF:: mean=" << ph_bf.mean() << ",\tmax=" << ph_bf.maxCoeff() << ",\tmin=" << ph_bf.minCoeff() << std::endl;
}

float DenseCRF::submodularF(std::vector<int> set){
	std::vector<std::vector <int>> labelSet(M_, std::vector<int>(N_));

	//get A_j = A \intersection V_j (V_j is ground set for j-th label)
        for(int j = 0; j < M_; j++){
            for(int i = 0; i < N_; i++){
			labelSet[j][i] = set[i*M_ + j]; 
		}
	}

	//unary costs	
	float unaryVal = 0;

	MatrixXf unary = unary_->get();
        for(int j = 0; j < M_; j++)
            for(int i = 0; i < N_; i++){
                    unaryVal = unaryVal + unary(j, i)*set[i*M_ + j];
            } 

	//pairwise costs
	float pairwiseVal = 0;
	float pairwiseValj = 0;	

	for(int j = 0; j < M_; j++){
		pairwiseValj = 0;
		for(int a = 0; a < N_; a++){
			for(int b = a; b < N_; b++){
				if((labelSet[j][a] == 1 && labelSet[j][b] == 0) || (labelSet[j][a] == 0 && labelSet[j][b] == 1)){
				pairwiseValj += 0.5*computeGaussianWeight(featureMat_.col(a), featureMat_.col(b));
				}
			}
		}
		pairwiseVal += pairwiseValj;
	}
	return unaryVal + pairwiseVal;
//	return unaryVal;
}



 
void DenseCRF::greedyAlgorithmBruteForce(MatrixXf &Qs, MatrixXf &Q){

	MatrixXf negGrad(M_, N_);
	getNegGradient(negGrad, Q);

	int len = M_ * N_;

	Map<VectorXf> w(negGrad.data(), negGrad.size());

	std::vector<float> wSorted;

	//copy (to keep original w intact)
	for(int i = 0; i < len; i++)
		wSorted.push_back(w[i]);

	//sort in descending order
        std::sort(wSorted.begin(), wSorted.begin() + len);
        std::reverse(wSorted.begin(), wSorted.begin() + len);
       
	std::vector<int> indexVec(len);
	size_t n(0);
        std::generate(begin(indexVec), end(indexVec), [&]{ return n++; });
        std::sort(begin(indexVec), end(indexVec), [&](int i1, int i2) { return w[i1] > w[i2]; } );

	std::vector<std::vector<int> > sigma;

	for(int j = 0; j < M_; j++){
		std::vector<int> sigmaj;
		for(int k = 0; k < len; k++){
			int index = indexVec[k];
			if(index % M_ == j)
				sigmaj.push_back(index); //indexing from 0 - (NL -1)
		}
		sigma.push_back(sigmaj);
	}

	MatrixXf unary  = unary_->get();
	MatrixXf Qs_pairwise = MatrixXf::Constant(M_, N_, 0);
	MatrixXf Qs_unary = MatrixXf::Constant(M_, N_, 0);

	for(int j = 0; j < M_; j++)
	for(int a = 0; a < N_; a++){
		int var_a = (int) sigma[j][a]/M_;
		VectorXf feature_a = featureMat_.col(var_a);
		Qs_unary(j, a) = unary(j, var_a);
		for(int b = 0; b < N_; b++){
			int var_b = (int) sigma[j][b]/M_;
			VectorXf feature_b = featureMat_.col(var_b);
			if(b > a)
				Qs_pairwise(j, a) += computeGaussianWeight(feature_a, feature_b)/2;
			else if (b < a)
				Qs_pairwise(j, a) += -computeGaussianWeight(feature_a, feature_b)/2;
			 else
				Qs_pairwise(j, a) += 0;
//			 std::cout << "j = " << j << " a = " << a << " b = " << b << " " << Qs_pairwise(j, a) << std::endl;
		}
	}

	for(int j = 0; j < M_; j++)
		for(int i = 0; i < N_; i++){
			int index = sigma[j][i];
			int row = index % M_;
			int col = index/M_;
			Qs(row, col) = Qs_pairwise(j, i) + Qs_unary(j, i);
		}

}

void DenseCRF::greedyAlgorithm(MatrixXf & Qs, MatrixXf &Q, int method){
	
	MatrixXf negGrad( M_, N_ );
	getNegGradient(negGrad, Q); //negative gradient

	Qs = MatrixXf::Zero(Qs.rows(), Qs.cols());	

	//easier to work with vector format of -ve gradient
	MatrixXf unaryOrdered(M_, N_), unary(M_, N_), temp2(M_, N_), rescaled_Q(M_, N_);
	MatrixXi ind(M_, N_);
	MatrixXf temp4 = MatrixXf::Constant(M_, N_, 0);
	MatrixXf temp3 = MatrixXf::Constant(M_, N_, 0);
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

	std::vector<std::vector<int> > sigma;

	for(int j = 0; j < M_; j++){
		std::vector<int> sigmaj;
		int countj = N_;
		for(int k = 0; k < len; k++){
			int index = indexVec[k];
			if(index % M_ == j){
				int i = (int) index/M_;		
				temp2(j, i) = countj; 
				countj = countj - 1;
				sigmaj.push_back(index); //indexing from 0 - (NL -1)
			}
		}
		sigma.push_back(sigmaj);
	}
		
	unary = unary_->get();

	for(int j = 0; j < M_; j++){
		for(int i = 0; i < N_; i++){
			unaryOrdered(j, i) = unary(j, (int) sigma[j][i]/M_);
		}
	}

	rescale(rescaled_Q, temp2);
        temp3 = unaryOrdered;
        sortRows(Q, ind);

	for( unsigned int k=0; k<pairwise_.size(); k++ ) {
		if(method == 1)
			pairwise_[k]->apply_upper_minus_lower_dc(temp4, ind);
		else if(method == 2)
			    pairwise_[k]->apply_upper_minus_lower_ord(temp4, rescaled_Q);
	    temp4 = temp4/2;
	    temp3 += temp4;
        }

	int currentNonZeroIndex = 0;

	VectorXf condGrad = VectorXf::Zero(Qs.rows() * Qs.cols());	
	for(unsigned int j=0; j<M_; j++ ) {
		VectorXf condGradj = VectorXf::Zero(Qs.rows() * Qs.cols());	
		for(unsigned int i=0; i<N_; i++ ) {
			condGradj[sigma[j][i]] = temp3(j, i);
		}
		condGrad += condGradj;
	}
		Map<MatrixXf> condGradMat(condGrad.data(), M_, N_);
		Qs = condGradMat;
}
