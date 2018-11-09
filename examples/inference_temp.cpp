#include <chrono>
#include <fstream>
#include <string>
#include <cstddef>
#include <vector>
#include "densecrf.h"
#include "file_storage.hpp"

void image_inference(std::string image_file, std::string unary_file, std::string dataset_name, 
            std::string method, std::string results_path, float spc_std, float spc_potts, 
            float bil_spcstd, float bil_colstd, float bil_potts, LP_inf_params & lp_params)
{
    img_size size = {-1, -1};

    MatrixXf unaries;    
    unsigned char * img;
      
    /*if rescaling*/
    int imskip = 1;
    if(dataset_name == "Denoising")
        img = load_grayscale_image(image_file, size, imskip);
    else
        img = load_rescaled_image(image_file, size, imskip);

    std::cout << "Image loaded" << std::endl;
      if(dataset_name == "MSRC"){
            unaries = load_unary_rescaled(unary_file, size, imskip);
        }else if (dataset_name == "Stereo_special"){
          unaries = load_unary_from_text(unary_file, size, imskip);
    } else if (dataset_name == "Denoising"){
          unaries = load_unary_from_text(unary_file, size, imskip);
    }
       
    size.height = size.height/imskip;
    size.width = size.width/imskip;

    DenseCRF2D crf(size.width, size.height, 256);
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
    crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,
                             bil_colstd, bil_colstd, bil_colstd,
                             img, new PottsCompatibility(bil_potts));

    crf.getFeatureMat(img);

    MatrixXf Q;
    std::size_t found = image_file.find_last_of("/\\");
    std::string image_name = image_file.substr(found+1);
    found = image_name.find_last_of(".");
    image_name = image_name.substr(0, found);
    image_name = image_name + "_" + std::to_string(spc_std) + "_" + std::to_string(spc_potts) + "_" + std::to_string(bil_spcstd) + "_" + std::to_string(bil_colstd) + "_" + std::to_string(bil_potts);
    std::string path_to_subexp_results = results_path + "/" + dataset_name + "_" + method + "_" + std::to_string(spc_std) + "_" + std::to_string(spc_potts) + "_" + std::to_string(bil_spcstd) + "_" + std::to_string(bil_colstd) + "_" + std::to_string(bil_potts) + "/";
    std::string output_path = get_output_path(path_to_subexp_results, image_name);
    make_dir(path_to_subexp_results);

    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;
    double timing;
    std::vector<int> pixel_ids;
    start = std::chrono::high_resolution_clock::now();

    Q = crf.unary_init();
    if (method == "mf5") {
        std::cout << "Starting mf inference " << std::endl;
        Q = crf.mf_inference(Q, 5, output_path, dataset_name);
     } else if (method == "submod") {
        std::cout << "Starting submod inference " << std::endl;
        Q = unaries;
        Q = crf.submodular_inference(Q, size.width, size.height, output_path, dataset_name);
    }   

    //save marginals
    std::string marg_file = output_path;
    marg_file.replace(marg_file.end()-4, marg_file.end(), "_marginals.txt");
    save_matrix(marg_file, Q, size);
     
    end = std::chrono::high_resolution_clock::now();
    timing = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();

    //save image
    save_map(Q, size, output_path, dataset_name);

    //calculate energy and print and save
    double discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));

    std::string txt_output = output_path;
    txt_output.replace(txt_output.end()-3, txt_output.end(),"txt");
    std::ofstream txt_file(txt_output);
    txt_file << timing << '\t' << discretized_energy << std::endl;
    std::cout << "#" << method << ": " << timing << '\t' << discretized_energy << std::endl;
    txt_file.close();
}

int main(int argc, char* argv[]) 
{
    if (argc < 5) {
        std::cout << "./example_inference image_file unary_file method results_path "
            "dataset_name spc_std spc_potts bil_spcstd bil_colstd bil_potts " << std::endl;
        std::cout << "Example: ./example_inference /path/to/image /path/to/unary "
            "[unary, mf5, mf, lrqp, qpcccp, dc-neg, sg-lp, cg-lp, prox-lp, prox-lp-l, prox-lp-acc] "
            "/path/to/results [MSRC, Pascal2010] [float] [float] [float] [float] [float] " << std::endl;
        return 1;
    }


    // set input, output paths and method
    std::string image_file = argv[1];
    std::string unary_file = argv[2];
    std::string method = argv[3];
    std::string results_path = argv[4];

    // set datasetname: MSRC or Pascal2010
    // default is MSRC, used to set energy parameters and color-map of the segmentation
    std::string dataset_name = "MSRC";
    if (argc > 5) {
        dataset_name = argv[5];
    }

    // set energy parameters: defaults to parameters tuned for dc-neg on MSRC dataset
    // cross-validated parameters for other methods can be found in data/cv-results.txt
    float spc_std = 3.535267;
    float spc_potts = 2.247081;
    float bil_spcstd = 31.232626;
    float bil_colstd = 7.949970;
    float bil_potts = 1.699011;

    if (argc < 11) { 
        if (dataset_name == "Pascal2010") {
            spc_std = 3.071772;
            spc_potts = 0.5;
            bil_spcstd = 49.78567;
            bil_colstd = 1;
            bil_potts = 0.960811;
        } else if (dataset_name != "MSRC") {
            dataset_name = "MSRC";
            std::cout << "Unrecognized dataset name, defaults to MSRC..." << std::endl;
        }         
    } else {
        spc_std = std::stof(argv[6]);
        spc_potts = std::stof(argv[7]);
        bil_spcstd = std::stof(argv[8]);
        bil_colstd = std::stof(argv[9]);
        bil_potts = std::stof(argv[10]);
    }

    std::cout << "#COMMAND: " << argv[0] << " " << image_file << " " << unary_file << " " << method << " " 
        << results_path << " " << dataset_name << " " << spc_std << " " << spc_potts << " " << bil_spcstd << " "
        << bil_colstd << " " << bil_potts << " " << std::endl;

   LP_inf_params lp_params;

    image_inference(image_file, unary_file, dataset_name, method, results_path, spc_std, spc_potts, 
            bil_spcstd, bil_colstd, bil_potts, lp_params);

    return 0;
}

