#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include "test.h" 
#include "densecrf.h"
#include <Eigen/Dense>
#include "file_storage.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;
using namespace Eigen;

std::string PATH_TO_RESULT = ".";
std::string PATH_TO_RESULT2 = ".";

int main(int argc, char* argv[])
{
   string dataPath = "../../data/";
   string image_file = dataPath + "2_14_s.bmp"; 
   string unary_file = dataPath + "2_14_s.c_unary"; 
   img_size size = {-1, -1};
   int imskip = 20;
   MatrixXf unaries = load_unary_rescaled(unary_file, size, imskip);
//   MatrixXf unaries = load_unary(unary_file, size);
    int N = unaries.cols(); 
    int M = unaries.rows(); 
    std::cout << size.width << " " << size.height << std::endl;

//	//save unaries to a file
//	ofstream unaryfile;
//	unaryfile.open ("unaries.txt");
//	unaryfile << unaries << std::endl;
//	unaryfile.close();
//
	MatrixXf negGrad(M, N);
	getNegGradient(negGrad, unaries);
	
//	//save negGrad to a file
//	ofstream gradfile;
//	gradfile.open ("gradient.txt");
//	gradfile << negGrad << std::endl;
//	gradfile.close();

   unsigned char * img = load_rescaled_image(image_file, size, imskip);

//	
//    //create CRF object
////    float spc_std = 3;
////    float spc_potts = 3;
////    float bil_spcstd = 60; 
////    float bil_colstd = 20;
////    float bil_potts = 10;
//
//    float spc_std = 1;
//    float spc_potts = 7.467846;
//    float bil_spcstd = 35.865959;
//    float bil_colstd = 11.209644;
//    float bil_potts = 4.028773;
    float spc_std = 0.5*1;
    float spc_potts = 0.5*7.467846;
    float bil_spcstd = 0.5*35.865959;
    float bil_colstd = 11.209644;
    float bil_potts = 4.028773;


    DenseCRF2D crf(size.width, size.height, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
    crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,
                             bil_colstd, bil_colstd, bil_colstd,
                             img, new PottsCompatibility(bil_potts));

    crf.getFeatureMat(img);
    MatrixXf feature = crf.getFeatureMat(img);
    MatrixXf out_bf(M, N), out_filter(M, N);
    MatrixP dot_tmp(M, N);

    float duration = 0;

    //MatrixXf negGrad = 10*(MatrixXf::Random(M, N).array() + 1);
      
    std::clock_t start = std::clock();
    crf.applyBruteForce(out_bf, negGrad);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout << "BF timing (s) = " << duration << std::endl;

    start = std::clock();
    crf.applyFilter(out_filter, negGrad);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout << "Filtering timing (s) = " << duration << std::endl;

    double costh = dotProduct(out_bf, out_filter, dot_tmp)/
       (sqrt(dotProduct(out_bf, out_bf, dot_tmp))*sqrt(dotProduct(out_filter, out_filter, dot_tmp)));
    std::cout << "bf-filter #cos-theta: " << costh << std::endl;

//    std::cout << "Qs_bf1 norm = " << Qs_bf1.norm() << std::endl;
 //   std::cout << "Qs_bf2 norm = " << Qs_bf2.norm() << std::endl;
 //   std::cout << "Qs_bfAJ norm = " << Qs_bfAJ.norm() << std::endl;
//    std::cout << "Difference norm = " << (Qs_bf - Qs_bfAJ).norm() << std::endl;
//

//    std::cout << "applyFilter() time = " << duration << std::endl;
//    std::cout << "Qs_bf norm = " << Qs_bf.norm() << std::endl;
//    std::cout << "Qs_filter norm = " << Qs_filter.norm() << std::endl;
//    std::cout << "Difference norm = " << (Qs_bf - Qs_filter).norm() << std::endl;

//j    MatrixXf Qs_exact(M, N), Qs_dc(M, N), Qs_filter(M, N);
//j
//j   crf.greedyAlgorithm(Qs_dc, Q, 1); 
//j   crf.greedyAlgorithm(Qs_filter, Q, 2); 
//j   crf.greedyAlgorithmBruteForce(Qs_exact, Q); 
//j
//j   Qs_dc = Qs_dc - Q;
//j   Qs_filter = Qs_filter - Q;
//j   Qs_exact = Qs_exact - Q;
//k
//j
//j	VectorXf dcVec(Map<VectorXf>(Qs_dc.data(), Qs_dc.cols()*Qs_dc.rows()));
//j	VectorXf filterVec(Map<VectorXf>(Qs_filter.data(), Qs_filter.cols()*Qs_filter.rows()));
//j	VectorXf exactVec(Map<VectorXf>(Qs_exact.data(), Qs_exact.cols()*Qs_exact.rows()));
//j	
//j	float angle_dc = acos(dcVec.dot(exactVec)/(dcVec.norm()*exactVec.norm()));
//j	float angle_filter = acos(filterVec.dot(exactVec)/(filterVec.norm()*exactVec.norm()));
//j	float angle_dc_filter = acos(filterVec.dot(dcVec)/(filterVec.norm()*dcVec.norm()));
//j	std::cout << "Dc-exact angle = " << angle_dc << std::endl; 
//j	std::cout << "filter-exact angle = " << angle_filter << std::endl; 
//j	std::cout << "Dc-filter angle = " << angle_dc_filter << std::endl; 



//j	VectorXf condGradApproxVec(Map<VectorXf>(Qs_approx.data(), Qs_approx.cols()*Qs_approx.rows()));
//j	float normApprox = (condGradApproxVec - Q).norm();
//j	std::cout << "Norm approx = " << normApprox << std::endl;
//j	
//j	crf.greedyAlgorithmBruteForce(Qs_exact, negGrad);
//j	VectorXf condGradExactVec(Map<VectorXf>(Qs_exact.data(), Qs_exact.cols()*Qs_exact.rows()));
//j	float normExact = (condGradExactVec - Q).norm();
//j	std::cout << "Norm exact = " << normExact << std::endl;
//j
//j	float dotProd = (condGradApproxVec - Q).dot(condGradExactVec - Q);
//j	float angle = acos(dotProd/(normApprox*normExact))*180/M_PI;
//j	std::cout << "Angle (degree)= " << angle << std::endl;
//j	std::cout << "Dot product = " << dotProd << std::endl;

}
