#ifndef FILE_STORAGE_H
#define FILE_STORAGE_H


#include <Eigen/Core>
#include "color_to_label.hpp"
#include <vector>

using namespace Eigen;

struct inputParameters{
    int nvar;
    int nlabel;
    double M;
    int m;
    MatrixXf unaries; 
    int nclique;
    std::vector< std::vector<int> > clique_members;
    std::vector< std::vector<int> > variable_clique_id; //which cliques does a variables belong to?
    std::vector<double> clique_weight;
    std::vector<int> clique_sizes;
};

struct img_size {
    int width;
    int height;
};

const unsigned char toy_legend[2*3] = {
    128, 0, 0,
    0, 128, 0
};

const unsigned char MSRC_legend[22*3] = {
    128,0,0,
    0,128,0,
    128,128,0,
    0,0,128,
    //horses are ignored 128,0,128,
    0,128,128,
    128,128,128,
    //mountains are also ignored 64,0,0,
    192,0,0,
    64,128,0,
    192,128,0,
    64,0,128,
    192,0,128,
    64,128,128,
    192,128,128,
    0,64,0,
    128,64,0,
    0,192,0,
    128,64,128,
    0,192,128,
    128,192,128,
    64,64,0,
    192,64,0,
    0,0,0
};


const unsigned char Pascal_legend[21*3] = {
    0,0,0,
    128,0,0,
    0,128,0,
    128,128,0,
    0,0,128,
    128,0,128,
    0,128,128,
    128,128,128,
    64,0,0,
    192,0,0,
    64,128,0,
    192,128,0,
    64,0,128,
    192,0,128,
    64,128,128,
    192,128,128,
    0,64,0,
    128,64,0,
    0,192,0,
    128,192,0,
    0,64,128,
};
const unsigned char Stereo_legend[16*3] = {
    0,0,0,
    15,15,15,
    30,30,30,
    45,45,45,
    60,60,60,
    75,75,75,
    90,90,90,
    105,105,105,
    120,120,120,
    135,135,135,
    150,150,150,
    165,165,165,
    180,180,180,
    195,195,195,
    210,210,210,
    225,225,225,
};

class Dataset {
protected:
    std::string path_to_images, path_to_unaries, path_to_ground_truths, path_to_root;
    std::string image_format, ground_truth_format;
public:
    std::string name;
    Dataset(std::string path_to_images, std::string path_to_unaries,
            std::string path_to_ground_truths, std::string path_to_root,
            std::string image_format, std::string ground_truth_format,
            std::string name);
    std::string get_unaries_path(const std::string & image_name);
    std::string get_ground_truth_path(const std::string & image_name);
    std::string get_image_path(const std::string & image_name);
    std::vector<std::string> get_all_split_files(const std::string & split);
};


Dataset get_dataset_by_name(const std::string & dataset_name);

bool file_exist(std::string file_path);
void make_dir(std::string dir_path);
void split_string(const std::string &s, const char delim, std::vector<std::string> &elems);

std::string get_output_path(const std::string & path_to_results_folder, const std::string & image_name);

unsigned char* load_image(const std::string& path_to_image, img_size & size);
unsigned char* load_rescaled_image(const std::string& path_to_image, img_size & size, int imskip = 1);
Matrix<short,Dynamic,1> load_labeling(const std::string & path_to_labels, const std::string & dataset_name,
                                      img_size& size);
MatrixXf load_unary(const std::string & path_to_unary, img_size& size, int max_label=-1);
MatrixXf load_unary_from_text(const std::string & path_to_unary, img_size& size, int imskip);
MatrixXf load_unary_rescaled(const std::string & path_to_unary, img_size& size, int imskip, int max_label = -1);
void load_unary_synthetic(const std::string file_path, int nvar, int nlabel, MatrixXf &unaries);
inputParameters load_unary_cliques(const std::string & fileName);
void save_map(const MatrixXf & estimates, const img_size &  size,
              const std::string & path_to_output, const std::string & dataset_name);
unsigned char *load_grayscale_image(const std::string & path_to_image, img_size & size, int imskip = 1);

void save_less_confident_pixels(const MatrixXf & estimates, const std::vector<int> & pI, const img_size &  size,
              const std::string & path_to_output, const std::string & dataset_name);

label_matrix load_label_matrix(const std::string & path_to_labels, const std::string & dataset_name);
label_matrix get_label_matrix(const MatrixXf & estimates, const img_size & size);


void save_image(unsigned char * img, const img_size & size, const std::string & path_to_output);
MatrixXf load_matrix(std::string path_to_matrix);
void save_matrix(std::string path_to_output, MatrixXf matrix, const img_size & size);
namespace Eigen{
void write_binary(std::string file_output, const MatrixXf& matrix);
}


#endif /* FILE_STORAGE_H */
