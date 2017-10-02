#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <ctime>
#include <math.h> 
#include "densecrf_utils.h"
#include "permutohedral.h"
#include "pairwise.h"
#include "file_storage.hpp"
#include "sparsecrf.h"

struct Neighbor{
    int n1;
    int n2;
    int n3;
    int n4;
};

/////////////////////////////
/////  Alloc / Dealloc  /////
/////////////////////////////

SparseCRF::SparseCRF(int W, int H, int M): W_(W), H_(H), M_(M), N_(H*W){
    unary_ = Eigen::MatrixXf::Zero(M_, N_);
    pairwise_weight_ = Eigen::VectorXf::Zero(M_, N_); 
}

//////////////////////////////
/////  Unary Potentials  /////
//////////////////////////////


void SparseCRF::readUnary(std::string file, int rows, int cols) {

  std::ifstream in(file);
  
  std::string line;

  int row = 0;
  int col = 0;

  if (in.is_open()) {

    while (std::getline(in, line)) {

      char *ptr = (char *) line.c_str();
      int len = line.length();

      col = 0;

      char *start = ptr;
      for (int i = 0; i < len; i++) {

        if (ptr[i] == ',') {
          unary_(row, col++) = atof(start);
          start = ptr + i + 1;
        }
      }
      unary_(row, col) = atof(start);

      row++;
    }

    in.close();
  }
}

Eigen::MatrixXf SparseCRF::getUnary() {
    return unary_;
}

void SparseCRF::setUnary(Eigen::MatrixXf unary) {
    unary_ = unary;
}
/////////////////////////////////
/////  Pairwise Potentials  /////
/////////////////////////////////

void SparseCRF::setPottsWeight(float weight) {
    for(int i = 0; i < M_; i++){
        pairwise_weight_(i) = weight;
    }
}

/////////////////////////////////////////
void SparseCRF::getNeighbors(int var, int grid_size, int *neighbor){ //there are grid_size*grid_size variables

    int var_x = var/grid_size; //row number for elem in the N X N grid
    int var_y = var % grid_size; //column number for elem

    std::fill_n(neighbor, 4, -1);

    if(var_x > 0)
        neighbor[0] = (var_x - 1)*grid_size + (var_y);
    if(var_x < grid_size - 1)
        neighbor[1] = (var_x + 1)*grid_size + (var_y);
    if(var_y > 0)
        neighbor[2] = (var_x)*grid_size + (var_y - 1);
    if(var_y < grid_size - 1)
        neighbor[3] = (var_x)*grid_size + (var_y + 1);

}

float SparseCRF::gridEnergyChange(int var, std::vector<int> S, int grid_size, int label){ //elem is between 0 - (NL - 1), N = grid_size ^ 2
    
   
    float unary_val = unary_(label, var);
    
    //get pairwise change
    float pairwise_val = 0;
    int neighbor[4];
    getNeighbors(var, grid_size, neighbor);
  
    for(int i = 0; i < 4; i++){
        if(neighbor[i] != -1){
            if(find(S.begin(), S.end(), neighbor[i]) != S.end()){
               pairwise_val = pairwise_val - 0.5*pairwise_weight_(label); 
            }
            else{
               pairwise_val =  pairwise_val + 0.5*pairwise_weight_(label); 
            }
        }
    }
    return unary_val + pairwise_val;
}

void SparseCRF::greedyAlgorithm(MatrixXf &out, MatrixXf &grad, int grid_size){

    //negative gradient at current point is input
    //LP solution is the output
    
//    cout << "Sorted grad" << endl;
    for(int j = 0; j < grad.rows(); j++){
        VectorXf grad_j = grad.row(j);

        std::vector<int> y(grad_j.size());
        iota(y.begin(), y.end(), 0);
        auto comparator = [&grad_j](int a, int b){ return grad_j[a] > grad_j[b]; };
        sort(y.begin(), y.end(), comparator);

        std::sort(grad_j.data(), grad_j.data() + grad_j.size(), std::greater<float>()); 
//        cout << grad_j << endl << endl;
        std::vector<int> S = {};
        for(int i = 0; i < y.size(); i++){
            out(j, y[i]) = gridEnergyChange(y[i], S, grid_size, j);
            S.push_back(y[i]);
        }
    }
//    cout << endl << "conditional gradient = " << out << endl;
}

void SparseCRF::getConditionalGradient(MatrixXf &Qs, MatrixXf & Q, int grid_size){
    //current solution is the input matrix (in)
    //conditional gradient is output

	MatrixXf negGrad( M_, N_ );
	getNegGradient(negGrad, Q); //negative gradient

        Qs.fill(0);
	greedyAlgorithm(Qs, negGrad, grid_size);	
}

void SparseCRF::submodularFrankWolfe(MatrixXf & init, int grid_size, std::string log_filename){

    //clock
    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;
    double timing;
    start = std::chrono::high_resolution_clock::now();

    MatrixXf Q( M_, N_ ), Qs( M_, N_); //Q is the current point, Qs is the conditional gradient

    MatrixXf Qs_bf = MatrixXf::Zero(M_, N_);
    MatrixXf negGrad = MatrixXf::Zero( M_, N_ );

    Q = init;	//initialize to unaries
    
    //log file
    std::ofstream logFile;
    logFile.open(log_filename);

    float step = 0;
    
    float objVal = 0;

    objVal = getObj(Q);

    logFile << "0 " << objVal << " " << step << std::endl;

    for(int k = 1; k <= 100; k++){

      getNegGradient(negGrad, Q); //negative gradient

      getConditionalGradient(Qs, Q, grid_size);

      float fenchelGap = (Qs - Q).cwiseProduct(negGrad).sum();
    
//      step = 2.0/(k + 2);
      step = doLineSearch(Qs, Q);

      Q = Q + step*(Qs - Q); 

      objVal = getObj(Q);

    end = std::chrono::high_resolution_clock::now();
    timing = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
    
      logFile << timing << '\t' << objVal << '\t' << step << std::endl;
 //     std::cout << "Iter: " << k << " Obj = " << objVal << " Step size = " << step << " Gap = " << fenchelGap << std::endl;
      std::cout << "Iter: " << k << " Obj = " << objVal << " Step size = " << step << " Time = " << timing << " Gap = " << fenchelGap << std::endl;
        if(fenchelGap < 1)
            break;
    }
    std::cout << "Upper bound = " << objVal << std::endl;
}
