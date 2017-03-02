#include "densecrf_utils.h"
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

string PATH_TO_RESULT = "";
int main(int argc, char* argv[]){

	PATH_TO_RESULT = argv[1];
//	MatrixXf Q = MatrixXf::Constant(3, 4, 0);
	MatrixXf Q = MatrixXf::Random(3, 4);
//	MatrixXf negGrad = MatrixXf::Constant(3, 4, 0);
	MatrixXf negGrad = MatrixXf::Random(3, 4);
	float step = 1;	

	ofstream logFile;
	string logFilename = PATH_TO_RESULT + "log.txt";
	logFile.open(logFilename);
	logFile << getObj(Q) << endl;

	cout << "hello" << endl;
	for(int i = 0; i < 1000; i++){
		step = (float) 1;
		getNegGradient(negGrad, Q);
		Q = Q + step*negGrad;
		logFile << getObj(Q) << endl;
	}
}
