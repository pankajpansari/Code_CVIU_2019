#include <fstream>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <math.h>
#include <vector>
#include <set>
#include <Eigen>
#include <string>
#include <sys/stat.h>

using namespace std;
using namespace Eigen; 

struct Neighbor{
    int n1;
    int n2;
    int n3;
    int n4;
};

MatrixXf readCSV(string file, int rows, int cols);

void getNeighbors(int var, int grid_size, int *neighbor){ //there are grid_size*grid_size variables

    int var_x = var/grid_size; //row number for elem in the N X N grid
    int var_y = var % grid_size; //column number for elem

    fill_n(neighbor, 4, -1);

    if(var_x > 0)
        neighbor[0] = (var_x - 1)*grid_size + (var_y);
    if(var_x < grid_size - 1)
        neighbor[1] = (var_x + 1)*grid_size + (var_y);
    if(var_y > 0)
        neighbor[2] = (var_x)*grid_size + (var_y - 1);
    if(var_y < grid_size - 1)
        neighbor[3] = (var_x)*grid_size + (var_y + 1);

}

float grid_energy_change(int var, vector<int> S, int grid_size, Eigen::MatrixXf &unary, int label){ //elem is between 0 - (NL - 1), N = grid_size ^ 2
    
//    int var = elem % (grid_size*grid_size); //var from 0 - (N - 1)
////    cout << "elem = " << elem << " grid_size^2 = " << grid_size*grid_size << endl;
//    int label = elem / (grid_size*grid_size); //label from 0 - (L - 1)
 //  cout << "var = " << var << " label = " << label << endl;

    //get unary change
    
    float unary_val = unary(label, var);
//    float unary_val = 0;
    
    //get pairwise change
    float pairwise_val = 0;
    int neighbor[4];
    getNeighbors(var, grid_size, neighbor);
  
    for(int i = 0; i < 4; i++){
        if(neighbor[i] != -1){
//            int n_elem = label*(grid_size*grid_size) + neighbor[i];
//            cout << neighbor[i] << " " << n_elem << endl;
            if(find(S.begin(), S.end(), neighbor[i]) != S.end()){
               pairwise_val = pairwise_val - 0.5; 
            }
            else{
               pairwise_val =  pairwise_val + 0.5; 
            }
        }
    }
//    cout << "unary change = " << unary_val << " pairwise change = " << pairwise_val << endl;
    return unary_val + pairwise_val;
}

float getTruePartition(Eigen::MatrixXf &unary){
    int N = unary.cols();
    int nlabel = unary.rows(); 
    int grid_size = (int) sqrt(N); 

    for(int t = 0; t < pow(2, N); t++){
        vector<int> labeling(N);
        fill(labeling.begin(), labeling.end(), 0);
        float energy = 0;       
        int temp = t;
        for(int k = 0; k < N; k++){
            labeling[k] = temp % 2;
            temp = (int) temp/2; 
        }

        for(int j = 0; j < nlabel; j++){
            vector<int> S = {};
            for(int i = 0; i < labeling.size(); i++){
                if(labeling[i] == j){
                    energy +=  grid_energy_change(i, S, grid_size, unary, j);
                    S.push_back(i);
                }
            }
        }
        for(int p = 0; p < labeling.size(); p++) cout << labeling[p] << " ";
       cout << endl << "energy = " << energy << endl;
    }
}

void greedyAlgorithmSparse(MatrixXf &out, MatrixXf &grad, int grid_size, Eigen::MatrixXf &unary){

    //negative gradient at current point is input
    //LP solution is the output
    
//    cout << "Sorted grad" << endl;
    for(int j = 0; j < grad.rows(); j++){
        VectorXf grad_j = grad.row(j);

        vector<int> y(grad_j.size());
        iota(y.begin(), y.end(), 0);
        auto comparator = [&grad_j](int a, int b){ return grad_j[a] > grad_j[b]; };
        sort(y.begin(), y.end(), comparator);

        sort(grad_j.data(), grad_j.data() + grad_j.size(), greater<float>()); 
//        cout << grad_j << endl << endl;
        vector<int> S = {};
        for(int i = 0; i < y.size(); i++){
            out(j, y[i]) = grid_energy_change(y[i], S, grid_size, unary, j);
            S.push_back(y[i]);
        }
    }
//    cout << endl << "conditional gradient = " << out << endl;
}

void getNegGradient(MatrixXf & negGrad, const MatrixXf & Q){

//    int M_ = Q.rows();
    int N_ = Q.cols(); 
    float gradSum = 0;
    for(int i = 0; i < N_; i++){
        gradSum = 0;
        VectorXf b = Q.col(i);
        float minMarginal = b.minCoeff();
        b.array() -= minMarginal;
        b = (-b.array()).exp();
        gradSum = b.sum();
        b = b/gradSum;
        negGrad.col(i) = b;
    }
//    std::cout << "Neg grad in non tree = " << std::endl << negGrad;
}

void getConditionalGradient(MatrixXf &Qs, MatrixXf & Q, int grid_size, Eigen::MatrixXf &unary){
    //current solution is the input matrix (in)
    //conditional gradient is output

        int M = Q.rows();
        int N = Q.cols();

	MatrixXf negGrad( M, N );
	getNegGradient(negGrad, Q); //negative gradient

        Qs.fill(0);
	greedyAlgorithmSparse(Qs, negGrad, grid_size, unary);	
}


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

void submodular_inference(MatrixXf & init, int grid_size, Eigen::MatrixXf &unary){

    int M_ = init.rows();
    int N_ = init.cols();

    MatrixXf Q( M_, N_ ), Qs( M_, N_); //Q is the current point, Qs is the conditional gradient

    MatrixXf Qs_bf = MatrixXf::Zero(M_, N_);
    MatrixXf negGrad = MatrixXf::Zero( M_, N_ );

    Q = init;	//initialize to unaries
    
    //log file
    std::ofstream logFile;
    logFile.open("/home/pankaj/SubmodularInference/data/working/07_09_2017/log.txt");


    float step = 0.001;
    
    float objVal = 0;

    objVal = getObj(Q);

    logFile << "0 " << objVal << " " << step << std::endl;

    for(int k = 1; k <= 100; k++){

      getConditionalGradient(Qs, Q, grid_size, unary);

      step = 2.0/(k + 2);

//        cout << "conditional grad = " << endl << Qs << endl;
      Q = Q + step*(Qs - Q); 

      objVal = getObj(Q);

      logFile << k << " " << objVal << " " << step << std::endl;
      cout << "Iter: " << k << " Obj = " << objVal << " Step size = " << step << endl;

    }
    cout << "Upper bound = " << objVal << endl;
}

int main(){
    int N = 100*100; 
    int M = 20;
    MatrixXf unary = readCSV("/home/pankaj/SubmodularInference/data/working/11_09_2017/unary.txt", M, N);
    int grid_size = (int) sqrt(N); 
//    cout << unary << endl;
    submodular_inference(unary, grid_size, unary);
//    float energy = grid_energy_change(1, {}, grid_size, unary, 0) + grid_energy_change(2, {1}, grid_size, unary, 0) + grid_energy_change(0, {}, grid_size, unary, 1) + grid_energy_change(3, {0}, grid_size, unary, 1);
//    getTruePartition(unary);
}
