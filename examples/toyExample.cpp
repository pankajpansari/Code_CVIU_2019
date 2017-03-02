#include "densecrf.h"
#include <iostream>
#include <fstream>
//#include "file_storage.hpp"


//string PATH_TO_RESULT = "";
using namespace std;

int main(int argc, char* argv[]) 
{

    //get the input arguments
    
	Eigen::MatrixXf a(3, 2);
	a << 5, 2, 1, 6, 3, 8;
//	cout << a(2, 0) << endl;
//	cout << a(0, 1) << endl;
	cout << getObj(a) << endl;	

//	a << 5, 6, 2, 3, 1, 8;
//	a << 0, 0, 0, 0, 0 , 0;
//	a.transposeInPlace();
//	cout << getObj(a) << endl;	
//	cout << 2*log(3) << endl;
//	cout << exp(3) << endl;

}
