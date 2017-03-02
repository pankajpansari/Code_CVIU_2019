#include <fstream>
#include <iostream>
#include <Eigen/Eigenvalues>
#include <cmath>

using namespace std;
using namespace Eigen;

int main()
{
    MatrixXf a = MatrixXf::Random(3, 4);
    cout << a << endl;
    a.transposeInPlace();
    Map<VectorXf> v(a.data(), a.size());
    cout << v << endl;
    cout << "Hello world" << endl;
}
