#include <chrono>
#include <fstream>
#include <string>
#include <cstddef>
#include <vector>
#include "densecrf.h"
#include "file_storage.hpp"
#include "tree_utils.h"

using namespace Eigen;

void image_inference(std::string image_file, std::string unary_file, std::string tree_file, std::string dataset_name, std::string method, std::string results_path, float spc_std, float spc_potts, 
            float bil_spcstd, float bil_colstd, float bil_potts)
{
    img_size size = {-1, -1};

       MatrixXf unaries;    
       unsigned char * img;
      
    /*if rescaling*/
    int imskip = 1;
    img = load_rescaled_image(image_file, size, imskip);
    std::cout << "Image loaded" << std::endl;
      if(dataset_name == "MSRC"){
            unaries = load_unary_rescaled(unary_file, size, imskip);
        }else if (dataset_name == "Stereo_special" || dataset_name == "Denoising"){
          unaries = load_unary_from_text(unary_file, size, imskip);
    }
    
    size.height = size.height/imskip;
    size.width = size.width/imskip;

    getdim(unaries);
   
    std::cout << "Unaries max = " << unaries.maxCoeff() << "    min = " << unaries.minCoeff() << std::endl;
    std::cout << "Unaries size = " << unaries.rows() << "x" << unaries.cols() << std::endl; 
    std::cout << "Unaries loaded" << std::endl;

    DenseCRF2D crf(size.width, size.height, unaries.rows());
    crf.setUnaryEnergy(unaries);
    if(method == "mf_tree"){
        std::vector<node> G = readTree(tree_file);
        std::cout << "using mf tree" << std::endl;
        MatrixXf m = getPairwiseTable(G);
        crf.addPairwiseGaussian(spc_std, spc_std, new MatrixCompatibility(m.array()*spc_potts));
        crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,
                             bil_colstd, bil_colstd, bil_colstd,
                             img, new MatrixCompatibility(m.array()*bil_potts));
    } else{
        crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
        crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,
                             bil_colstd, bil_colstd, bil_colstd,
                             img, new PottsCompatibility(bil_potts));
    }
    MatrixXf Q;
    std::size_t found = image_file.find_last_of("/\\");
    std::string image_name = image_file.substr(found+1);
    found = image_name.find_last_of(".");
    image_name = image_name.substr(0, found);
    image_name = image_name + "_" + method + "_" + std::to_string(spc_std) + "_" + std::to_string(spc_potts) + "_" + std::to_string(bil_spcstd) + "_" + std::to_string(bil_colstd) + "_" + std::to_string(bil_potts);
    std::string path_to_subexp_results = results_path + "/" + dataset_name + "_" + method + "_" + std::to_string(spc_std) + "_" + std::to_string(spc_potts) + "_" + std::to_string(bil_spcstd) + "_" + std::to_string(bil_colstd) + "_" + std::to_string(bil_potts) + "/";
    std::string output_path = get_output_path(path_to_subexp_results, image_name);
    make_dir(path_to_subexp_results);

    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;
    double timing;
    std::vector<int> pixel_ids;
    start = std::chrono::high_resolution_clock::now();

    Q = crf.unary_init();

    double initial_discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));
    std::cout << "Initial energy = " << initial_discretized_energy << std::endl;


    if (method == "mf5") {
        std::cout << "Starting mf inference " << std::endl;
        Q = crf.mf_inference(Q, 5, output_path, dataset_name);
    } else if (method == "mf" || method == "mf_tree") {
        std::cout << "Starting mf inference " << std::endl;
        Q = crf.mf_inference(Q, output_path, dataset_name);
    } else if (method == "submod") {
        std::cout << "Starting submod inference " << std::endl;
        Q = unaries;
        Q = crf.submodularFrankWolfe_Potts(Q, size.width, size.height, output_path, dataset_name);
   } else if (method == "unary") {
        (void)0;
    } else {
        std::cout << "Unrecognised method: " << method << ", exiting..." << std::endl;
        return;
    }
     
    //get timing
    end = std::chrono::high_resolution_clock::now();
    timing = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();

    //get energy 
   double discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));

   //write energy, time to text file
    std::string txt_output = output_path;
    txt_output.replace(txt_output.end()-3, txt_output.end(),"txt");
    std::ofstream txt_file(txt_output);
    txt_file << timing << '\t' << discretized_energy << std::endl;
    std::cout << "#" << method << ": " << timing << '\t' << discretized_energy << std::endl;
    txt_file.close();

    //save map
    save_map(Q, size, output_path, dataset_name);

    //save marginals
   std::string marg_file = output_path;
   marg_file.replace(marg_file.end()-4, marg_file.end(), "_marginals.txt");
   save_matrix(marg_file, Q, size);
 
}

int main(int argc, char* argv[]) 
{
    // set input, output paths and method
    assert(argc == 12 && "All arguments not given");
    std::string image_file = argv[1];
    std::string unary_file = argv[2];
    std::string tree_file = argv[3];
    std::string method = argv[4];
    std::string results_path = argv[5];

    std::string dataset_name = argv[6];

     float  spc_std = std::stof(argv[7]);
     float  spc_potts = std::stof(argv[8]);
     float  bil_spcstd = std::stof(argv[9]);
     float  bil_colstd = std::stof(argv[10]);
     float  bil_potts = std::stof(argv[11]);


    std::cout << "#COMMAND: " << argv[0] << " " << image_file << " " << unary_file << " " << method << " " 
        << results_path << " " << dataset_name << " " << spc_std << " " << spc_potts << " " << bil_spcstd << " "
        << bil_colstd << " " << bil_potts << " " << std::endl;

   LP_inf_params lp_params;

   image_inference(image_file, unary_file, tree_file, dataset_name, method, results_path, spc_std, spc_potts, 
            bil_spcstd, bil_colstd, bil_potts);

    return 0;

}

