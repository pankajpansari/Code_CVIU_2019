#include <fstream>
#include <iostream>
#include <Eigen/Eigenvalues>
#include <cmath>

using namespace Eigen;

float getObj(const MatrixXf & Q);
void getNegGradient(MatrixXf & negGrad, const MatrixXf & Q);
void getNegGradient2(MatrixXf & negGrad, const MatrixXf & Q);
