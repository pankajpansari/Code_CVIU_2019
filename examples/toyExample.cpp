#include "densecrf.h"
#include "pairwise.h"
#include <iostream>
#include <fstream>
//#include "file_storage.hpp"
//g++ -std=c++11 -I ../include/ toyExample.cpp

//string PATH_TO_RESULT = "";
using namespace std;
using namespace Eigen; 

float getObj(const MatrixXf & Q){

	float logSum = 0, expSum = 0, minMarginal = 0;

	for(int i = 0; i < Q.cols(); i++){
            expSum = 0;
            VectorXf b = Q.col(i);
            minMarginal = b.minCoeff();
            b.array() -= minMarginal; 
            b = (-b.array()).exp(); 
            expSum = b.sum();
            logSum = logSum + log(expSum) - minMarginal;
        }	

	return logSum;
}

float getDeriv(const MatrixXf & Q, const MatrixXf & Qs, float step){

        float deriv = 0; 
	for(int i = 0; i < Q.cols(); i++){
            VectorXf a = Q.col(i);
            a = (-a.array()).exp();
            VectorXf b = Qs.col(i) - Q.col(i);
            VectorXf c = (-b.array()*step).exp();
            VectorXf temp1 = -a.array()*b.array()*c.array();
            VectorXf temp2 = a.array()*c.array();
            deriv = deriv + temp1.sum()/temp2.sum();  
        }
        return deriv;
} 

float doLineSearch2(const MatrixXf & Qs, const MatrixXf & Q){

	//do binary search for line search
	float rangeStart = 0;
	float rangeEnd = 1; 
        float middle = 0;
        float deriv = 0;

	for(int binaryIter = 0; binaryIter <= 20; binaryIter++){	
                
                middle = (rangeStart + rangeEnd)/2;
                deriv = getDeriv(Q, Qs, middle); 

                cout << middle << " " << deriv << endl;
                if(deriv > 0)
                    rangeEnd = middle;
                else if(deriv < 0)
                    rangeStart = middle;
                else
                    return middle;
	}

        return middle;
}

int main(int argc, char* argv[]) 
{

    int N = 10000, M = 21;
    MatrixXf Q = MatrixXf::Random(M, N);
    MatrixXf Qs = MatrixXf::Random(M, N);
    
    ofstream logFile;
    logFile.open("deriv.txt");
    for(float step = 0; step <= 1; step = step + 0.01){
        MatrixXf temp = Q + step*(Qs - Q);
        float objVal = getObj(temp);
        float deriv = getDeriv(Q, Qs, step);
        logFile << step << " " << objVal << " " << deriv << endl;
    }

    cout << "step = " << doLineSearch2(Qs, Q) << endl;
   //get the input arguments
    
//	MatrixXf a = MatrixXf::Constant(3, 2, 1);
        
//	a << 5, 2, 1, 6, 3, 8;
 //       a = a.array() - 1;
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
