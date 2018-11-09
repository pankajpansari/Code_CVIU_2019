//template
//compile & link - g++ trial.cpp
//execute	./a.out
#include <iostream>
#include <vector>

#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

void getNeighbors(int var, int grid_size, int *neighbor){ //there are grid_size*grid_size variables

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

float unaryChangeBadExtension(int var, int label, std::vector<int> S, MatrixXf &unary){
    std::vector<int> st_vec(unary.rows()); // 0 -> in s, 1 -> in t
    std::vector<int> temp(unary.rows());  //which unary costs to add
    float unary_cost = 0;
    for(int i = 0; i < unary.rows(); i++){
        st_vec[i] = 0;
        temp[i] = 0;
    }

   for(int i = 0; i < S.size(); i++){
       st_vec[S[i]] = 1;
   }
   st_vec[label] = 1;

   for(int i = 0; i < st_vec.size(); i++){
       if(i == st_vec.size() - 1){
        if(st_vec[i] != st_vec[0]){
           temp[i] = 1;
        }
       }
       else if(st_vec[i] != st_vec[i + 1])
           temp[i] = 1;
      }

   for(int i = 0; i < st_vec.size(); i++){
       if(i == st_vec.size() - 1){
        if(st_vec[i] != st_vec[0]){
           temp[i] = 1;
        }
       }
       else if(st_vec[i] != st_vec[i + 1])
           temp[i] = 1;
      }

for(int i = 0; i < temp.size(); i++){
        if(temp[i] == 1)
        unary_cost += unary(i, var);
}
   return unary_cost;
}

float gridEnergyChangeBadExtension(int var, std::vector<int> S, int grid_size, int label){ //elem is between 0 - (NL - 1), N = grid_size ^ 2
    
   
    //get pairwise change
    float pairwise_val = 0;
    int neighbor[4];
    getNeighbors(var, grid_size, neighbor);
  
    for(int i = 0; i < 4; i++){
        if(neighbor[i] != -1){
            if(find(S.begin(), S.end(), neighbor[i]) != S.end()){
               pairwise_val = pairwise_val - 0.5; //0.5 not required here because tree assumes that
            }
            else{
               pairwise_val =  pairwise_val + 0.5; 
            }
        }
    }
    return pairwise_val;
}
void greedyAlgorithmBadExtension(MatrixXf &out, MatrixXf &grad, int grid_size){

    int M_ = grad.rows();
    int N_ = grad.cols();

    MatrixXf unary_out = MatrixXf::Zero(M_, N_); 
    MatrixXf pairwise_out = MatrixXf::Zero(M_, N_);
    MatrixXf unary = MatrixXf::Random(M_, N_);
    std::cout << "unary = " << std::endl << unary << std::endl;

    //for unaries
    for(int j = 0; j < grad.cols(); j++){
        VectorXf grad_j = grad.col(j);

        std::vector<int> y(grad_j.size());
        iota(y.begin(), y.end(), 0);
        auto comparator = [&grad_j](int a, int b){ return grad_j[a] > grad_j[b]; };
        sort(y.begin(), y.end(), comparator);

        std::sort(grad_j.data(), grad_j.data() + grad_j.size(), std::greater<float>()); 
        std::vector<int> S = {};

        for(int i = 0; i < y.size(); i++){
            unary_out(y[i], j) = unaryChangeBadExtension(j, y[i], S, unary);
            S.push_back(y[i]);
        }
    }


    //for pairwise
    for(int j = 0; j < grad.rows(); j++){
        VectorXf grad_j = grad.row(j);

        std::vector<int> y(grad_j.size());
        iota(y.begin(), y.end(), 0);
        auto comparator = [&grad_j](int a, int b){ return grad_j[a] > grad_j[b]; };
        sort(y.begin(), y.end(), comparator);

        std::sort(grad_j.data(), grad_j.data() + grad_j.size(), std::greater<float>()); 
        std::vector<int> S = {};

        for(int i = 0; i < y.size(); i++){
            pairwise_out(j, y[i]) = gridEnergyChangeBadExtension(y[i], S, grid_size, j);
            S.push_back(y[i]);
        }
    }
 
    out = unary_out + pairwise_out;
}

int main()
{
    int grid_size = 2;
    MatrixXf grad = MatrixXf::Random(3, grid_size*grid_size);
    MatrixXf out = MatrixXf::Zero(3, grid_size*grid_size);

    std::cout << grad << std::endl;

    greedyAlgorithmBadExtension(out, grad, grid_size);
    std::cout << std::endl;
    std::cout << out << std::endl;
}
