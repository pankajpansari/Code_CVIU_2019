//template
//compile & link - g++ trial.cpp
//execute	./a.out
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include "test.h" 
#include "densecrf.h"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
 	VectorXf a = VectorXf::Random(9);
	cout << a << endl;
	Map<MatrixXf> A(a.data(), 3, 3);
	cout << A << endl;
}
