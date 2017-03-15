#include "densecrf.h"
#include "pairwise.h"
#include <iostream>
#include <fstream>
//#include "file_storage.hpp"
//g++ -std=c++11 -I ../include/ toyExample.cpp

//string PATH_TO_RESULT = "";
using namespace std;
using namespace Eigen; 

void read_binary(const char* filename, MatrixXf& matrix){
    std::ifstream in(filename,std::ios::in | std::ios::binary);
    typename MatrixXf::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename MatrixXf::Index));
    in.read((char*) (&cols),sizeof(typename MatrixXf::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename MatrixXf::Scalar) );
    in.close();
}

void increment(int a, int *b){
    b[a] += 1; 
}

int main(int argc, char* argv[]) 
{

    MatrixXf a;
    read_binary("Q_10.dat", a);
    //get the input arguments
    
//	MatrixXf a = MatrixXf::Constant(3, 2, 1);
        
//	a << 5, 2, 1, 6, 3, 8;
 //       a = a.array() - 1;
    cout << a.col(2) << endl;
//    cout << a.cols() << endl;
//
//	cout << a << endl;	
//        float* b = a.data();
//	cout << b[0] << endl;	
//	cout << b[1] << endl;	
//	cout << b[2] << endl;	
//	cout << b[3] << endl;	
//	cout << b[4] << endl;	
//	cout << b[5] << endl;	
//
//        float t = 0.452;
//        int bin = t/0.1;
//	cout << bin << endl;	
        
//	a << 5, 6, 2, 3, 1, 8;
//	a << 0, 0, 0, 0, 0 , 0;
//	a.transposeInPlace();
//	cout << getObj(a) << endl;	
//	cout << 2*log(3) << endl;
//	cout << exp(3) << endl;

//    int a[4][5]; 
//    memset(a, 0, sizeof(a));
//    for(int i = 0; i < 4; i ++)
//        for(int j = 0; j < 5; j ++)
//            cout << a[i][j] << " ";
//
//    cout << endl << endl;
//
//    increment(0, a[2]);
//    increment(0, a[2]);
//
//    for(int i = 0; i < 4; i ++)
//        for(int j = 0; j < 5; j ++)
//            cout << a[i][j] << " ";
//
}
