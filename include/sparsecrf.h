#include "densecrf_utils.h"
#include <vector>
class SparseCRF{

private:
    int N_, M_;
    int H_, W_;
    
    // Store the unary term
    Eigen::MatrixXf unary_;
    float pairwise_weight_;    
//    // Store all pairwise potentials

public:
    // Create a dense CRF model of size N with M labels
//    SparseCRF( int N, int M, int H, int W );
    SparseCRF(int W, int H, int M);
 
    void readUnary(std::string file, int rows, int cols);
    void setUnary(Eigen::MatrixXf unary);
    
    void setPottsWeight(float weight);

    void getNeighbors(int var, int grid_size, int *neighbor);
    
    float gridEnergyChange(int var, std::vector<int> S, int grid_size,  int label);
    
    void greedyAlgorithm(Eigen::MatrixXf &out, Eigen::MatrixXf &grad, int grid_size);
    
    void getConditionalGradient(Eigen::MatrixXf &Qs, Eigen::MatrixXf & Q, int grid_size);
    
    Eigen::MatrixXf getUnary();

    void submodularFrankWolfe(Eigen::MatrixXf & init, int grid_size, std::string log_filename);

};
