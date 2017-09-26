#include <chrono>
#include <fstream>
#include <string>
#include <cstddef>
#include <vector>
#include "densecrf.h"
#include "file_storage.hpp"
#undef NDEBUG
#include <assert.h>
const int USE_TREES = 0;
   
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
     
    std::ofstream unaryFile;
    unaryFile.open("/home/pankaj/SubmodularInference/data/working/08_09_2017/image_unary.txt");
    unaryFile << unaries << std::endl;
    unaryFile.close();

    size.height = size.height/imskip;
    size.width = size.width/imskip;

    getdim(unaries);
    /*if not rescaling
     img = load_image(image_file, size);
     std::cout << "Image loaded" << std::endl;


     if(dataset_name == "MSRC"){
       unaries = load_unary(unary_file, size);
     }else if (dataset_name == "Stereo_special"){
       unaries = load_unary_from_text(unary_file, size, 1);
     }*/
    
    std::cout << "Unaries max = " << unaries.maxCoeff() << "    min = " << unaries.minCoeff() << std::endl;
    std::cout << "Unaries size = " << unaries.rows() << "x" << unaries.cols() << std::endl; 
    std::cout << "Unaries loaded" << std::endl;


    // create densecrf object
//    std::string tree_filename = "/home/pankaj/SubmodularInference/data/input/tests/trees/tree_5_2.txt";
    int nMeta = 0, nLabel = 0;
    std::string tree_filename = "";

    if(USE_TREES == 1){

    std::ifstream treefile(tree_filename);
    std::string s;

    std::getline(treefile, s);
    std::istringstream ss(s);

    ss >> nMeta >> nLabel;

    } else{
        nLabel = unaries.rows();
        nMeta = nLabel;
    }

    DenseCRF2D crf(size.width, size.height, nMeta);
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

    //check unary_init function: initialise with marginals or with unaries?
//    Q = crf.unary_init();
    if (method == "mf5") {
        std::cout << "Starting mf inference " << std::endl;
        Q = crf.mf_inference(Q, 5, output_path, dataset_name);

    } else if (method == "mf") {
        std::cout << "Starting mf inference " << std::endl;
        Q = crf.mf_inference(Q, output_path, dataset_name);
    } else if (method == "submod") {
        std::cout << "Starting submodular inference (without trees)" << std::endl;
        Q = unaries;
        Q = crf.submodular_inference_dense(Q, size.width, size.height, output_path, dataset_name);
    } else if (method == "submod_rhst") {
        assert(USE_TREES == 1 && "Cannot run tree algorithm. USE_TREES constant should be set to 1 \n");
         std::cout << "Starting submodular tree inference" << std::endl;
         Q = unaries;
         Q = crf.submodular_inference_rhst(Q, size.width, size.height, output_path, tree_filename,  dataset_name);
    } else if (method == "lrqp") {
        Q = crf.qp_inference(Q);
    } else if (method == "qpcccp") {
        Q = crf.qp_inference(Q);
        Q = crf.qp_cccp_inference(Q);
    } else if (method == "dc-neg") {
        Q = crf.qp_inference(Q);
        Q = crf.concave_qp_cccp_inference(Q);
    } else if (method == "sg-lp") {
        Q = crf.qp_inference(Q);
        Q = crf.concave_qp_cccp_inference(Q);
        Q = crf.lp_inference(Q, false);
    } else if (method == "cg-lp") {
        Q = crf.qp_inference(Q);
        Q = crf.concave_qp_cccp_inference(Q);
        Q = crf.lp_inference(Q, true);
    } else if (method == "prox-lp") {    // standard prox_lp
        Q = crf.qp_inference(Q);
        Q = crf.concave_qp_cccp_inference(Q);
        Q = crf.lp_inference_prox(Q, lp_params);    
    } else if (method == "prox-lp-l") {  // standard prox_lp with limited labels
        Q = crf.qp_inference(Q);
        Q = crf.concave_qp_cccp_inference(Q);

        htime st = std::chrono::high_resolution_clock::now();
        std::vector<int> indices;
        get_limited_indices(Q, indices);
        if (indices.size() > 1) {
            MatrixXf runaries = get_restricted_matrix(unaries, indices);
            MatrixXf rQ = get_restricted_matrix(Q, indices);
            DenseCRF2D rcrf(size.width, size.height, runaries.rows());
            rcrf.setUnaryEnergy(runaries);
            rcrf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
            rcrf.addPairwiseBilateral(bil_spcstd, bil_spcstd,
                         bil_colstd, bil_colstd, bil_colstd,
                         img, new PottsCompatibility(bil_potts));
            htime et = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::duration<double>>(et-st).count();
    
            rQ = rcrf.lp_inference_prox(rQ, lp_params);
            
            Q = get_extended_matrix(rQ, indices, unaries.rows());
        }
    } else if (method == "prox-lp-p") {    // standard prox_lp with limited pixels
        Q = crf.qp_inference(Q);
        Q = crf.concave_qp_cccp_inference(Q);

        lp_params.less_confident_percent = 10;
        lp_params.confidence_tol = 0.95;
        Q = crf.lp_inference_prox(Q, lp_params);
    
        // lp inference params
        LP_inf_params lp_params_rest = lp_params;
        lp_params_rest.prox_reg_const = 0.001;
        Q = crf.lp_inference_prox_restricted(Q, lp_params_rest);
        less_confident_pixels(pixel_ids, Q, lp_params.confidence_tol);                    
        
    } else if (method == "prox-lp-acc") {    // fully accelerated prox_lp
        Q = crf.qp_inference(Q);
        Q = crf.concave_qp_cccp_inference(Q);

        htime st = std::chrono::high_resolution_clock::now();
        std::vector<int> indices;
        get_limited_indices(Q, indices);
        if (indices.size() > 1) {
            MatrixXf runaries = get_restricted_matrix(unaries, indices);
            MatrixXf rQ = get_restricted_matrix(Q, indices);
            DenseCRF2D rcrf(size.width, size.height, runaries.rows());
            rcrf.setUnaryEnergy(runaries);
            rcrf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
            rcrf.addPairwiseBilateral(bil_spcstd, bil_spcstd,
                         bil_colstd, bil_colstd, bil_colstd,
                         img, new PottsCompatibility(bil_potts));
            htime et = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::duration<double>>(et-st).count();
            //std::cout << "#rcrf construction: " << dt << " seconds" << std::endl;
    
            lp_params.less_confident_percent = 10;
            lp_params.confidence_tol = 0.95;
            rQ = rcrf.lp_inference_prox(rQ, lp_params);
    
            // lp inference params
            LP_inf_params lp_params_rest = lp_params;
            lp_params_rest.prox_reg_const = 0.001;
            rQ = rcrf.lp_inference_prox_restricted(rQ, lp_params_rest);
            less_confident_pixels(pixel_ids, rQ, lp_params.confidence_tol);                    
            
            Q = get_extended_matrix(rQ, indices, unaries.rows());
        }
    } else if (method == "unary") {
        (void)0;
    } else {
        std::cout << "Unrecognised method: " << method << ", exiting..." << std::endl;
        return;
    }

//    std::string marg_file = output_path;
//    marg_file.replace(marg_file.end()-4, marg_file.end(), "_marginals.txt");
//    save_matrix(marg_file, Q, size);
     
    end = std::chrono::high_resolution_clock::now();
    timing = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
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

//./inference ../../data/2_14_s.bmp ../../data/2_14_s.c_unary submod . MSRC 1
//./single_image_inference ../../data/2_14_s.bmp ../../data/2_14_s.c_unary mf . MSRC 1 7.467846 35.865959 11.209644 4.028773

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

    //write command to key file
//    std::size_t found = image_file.find_last_of("/\\");
//    std::string image_name = image_file.substr(found+1);
//    std::string path_to_subexp_results = results_path + "/" + method + "/";
//    std::string key_output = get_output_path(path_to_subexp_results, image_name);
////    std::string key_output = get_output_path(results_path + "/" + method + "/", image_file);
//    key_output.replace(key_output.end()-4, key_output.end(),"_key.txt");
//    std::cout << key_output << std::endl;
//    std::ofstream keyFile;
//    keyFile.open(key_output);
//
//    keyFile << "#COMMAND: " << argv[0] << " " << image_file << " " << unary_file << " " << method << " " 
//        << results_path << " " << dataset_name << " " << spc_std << " " << spc_potts << " " << bil_spcstd << " "
//        << bil_colstd << " " << bil_potts << " " << imskip << " " << std::endl;
//
//    keyFile.close();
    
    // set prox-lp parameters: defaults to the settings given in the paper
    // defaults parameters are given in densecrf_utils.cpp
    LP_inf_params lp_params;

    image_inference(image_file, unary_file, dataset_name, method, results_path, spc_std, spc_potts, 
            bil_spcstd, bil_colstd, bil_potts, lp_params);

    return 0;
}
