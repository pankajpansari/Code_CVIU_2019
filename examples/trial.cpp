//template
//compile & link - g++ trial.cpp
//execute	./a.out
#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include "test.h" 
#include "densecrf.h"
#include <Eigen/Dense>
#include <math.h>

using namespace std;
using namespace Eigen;

string PATH_TO_RESULT = "./";
string PATH_TO_RESULT2 = "./";

int main(int argc, char* argv[])
{

    string dataPath = "../../data/";
    //read image from a random img file 
    int width, height;

    ifstream imgf(dataPath + "img1.txt");
    imgf >> width >>  height;

	unsigned char * char_img = new unsigned char[width*height*3];
//        float * img = new float[width*height*3];
	    for (int j = 0; j < height; j++)
		for (int i = 0; i < width; i++)
			for (int k = 0; k < 3; k++){
                                int a;
			      imgf >> a;
                              char_img[(i+j*width)*3+k] = (unsigned char) a;}
	    imgf.close();
	//read a random unary file
       int M, N;

    ifstream unaryf(dataPath + "unary1.txt");
        unaryf >> M >>  N;
	assert(N == height * width);
	MatrixXf unaries(M, N); 
	    for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
		      unaryf >> unaries(i, j);       
    unaryf.close();
 
    //create CRF object
//    float spc_std = 3;
//    float spc_potts = 3;
//    float bil_spcstd = 60; 
//    float bil_colstd = 20;
//    float bil_potts = 10;

    float spc_std = 1;
    float spc_potts = 7.467846;
    float bil_spcstd = 35.865959;
    float bil_colstd = 11.209644;
    float bil_potts = 4.028773;


    DenseCRF2D crf(width, height, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
    crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,
                             bil_colstd, bil_colstd, bil_colstd,
                             char_img, new PottsCompatibility(bil_potts));

    MatrixXf feature = crf.getFeatureMat(char_img);
    MatrixXf Qs_bf1(M, N), Qs_bf2(M, N);
    crf.test_apply(Qs_bf1, unaries);
    crf.greedyAlgorithmBruteForce(Qs_bf2, unaries);
    std::cout << (Qs_bf1 - Qs_bf2).norm() << std::endl;
//j    //read random Q
//j    ifstream Qf(dataPath + "Q1.txt");
//j    Qf >> M >>  N;
//j
//j    MatrixXf Q(M, N), Qs_exact(M, N), Qs_approx(M, N);
//j    for (int i = 0; i < M; i++)
//j        for (int j = 0; j < N; j++)
//j              Qf >> Q(i, j);       
//j
//j    Qf.close();
//j    
//j    //get -ve gradient at Q
//j	MatrixXf negGrad(M, N);
//j	getNegGradient(negGrad, Q);
//j
//j	crf.greedyAlgorithmBruteForce(Qs_exact, negGrad);
//j	VectorXf condGradExactVec(Map<VectorXf>(Qs_exact.data(), Qs_exact.cols()*Qs_exact.rows()));
//j	float normExact = condGradExactVec.norm();
//j	std::cout << "Norm exact = " << normExact << std::endl;
//j	crf.greedyAlgorithm(Qs_approx, negGrad);
//j	VectorXf condGradApproxVec(Map<VectorXf>(Qs_approx.data(), Qs_approx.cols()*Qs_approx.rows()));
//j	float normApprox = condGradApproxVec.norm();
//j	std::cout << "Norm exact = " << normApprox << std::endl;
//j	float dotProd = condGradApproxVec.dot(condGradExactVec);
//j	float angle = acos(dotProd/(normApprox*normExact))*180/M_PI;
//j	std::cout << "Angle (degree)= " << angle << std::endl;
}
