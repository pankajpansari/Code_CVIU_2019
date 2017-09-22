#include <chrono>
#include <fstream>
#include <string>
#include <cstddef>
#include <vector>
#include "densecrf.h"
#include "file_storage.hpp"

void image_inference(Dataset dataset, std::string method, std::string path_to_results,
                     std::string image_name, float spc_std, float spc_potts, float bil_spcstd, float bil_colstd, float bil_potts, LP_inf_params & lp_params)
{

    std::string image_file = dataset.get_image_path(image_name);
    std::string unary_file = dataset.get_unaries_path(image_name);
    std::string dataset_name = dataset.name;

    img_size size = {-1, -1};

    MatrixXf unaries;    
    unsigned char * img;
      
    /*if rescaling*/
    int imskip = 1;
    img = load_rescaled_image(image_file, size, imskip);
    std::cout << "Image loaded" << std::endl;
      if(dataset_name == "MSRC"){
            unaries = load_unary_rescaled(unary_file, size, imskip);
        }else if (dataset_name == "Stereo_special"){
          unaries = load_unary_from_text(unary_file, size, imskip);
    }
       
    size.height = size.height/imskip;
    size.width = size.width/imskip;

    getdim(unaries);
   
    std::cout << "Unaries max = " << unaries.maxCoeff() << "    min = " << unaries.minCoeff() << std::endl;
    std::cout << "Unaries size = " << unaries.rows() << "x" << unaries.cols() << std::endl; 
    std::cout << "Unaries loaded" << std::endl;

    // create densecrf object
    std::string tree_filename = "";
    if (dataset_name == "MSRC"){
        //if MSRC, then use Potts tree
        tree_filename = "/home/pankaj/SubmodularInference/data/input/tests/trees/long/msrc_potts.txt";
   } else if (dataset_name == "Stereo_special"){
        //if stereo, specify the appropriate tree file
         tree_filename = "/home/pankaj/SubmodularInference/data/input/tests/trees/test/venus_l1_M10/tree_full.txt";
   }


    std::ifstream treefile(tree_filename);
    std::string s;

    std::getline(treefile, s);
    std::istringstream ss(s);

    int nMeta, nLabel;
    ss >> nMeta >> nLabel;

//    DenseCRF2D crf(size.width, size.height, nMeta);
    std::cout << "nLabel = " << nLabel << std::endl;
    DenseCRF2D crf(size.width, size.height, nLabel);
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
    crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,
                             bil_colstd, bil_colstd, bil_colstd,
                             img, new PottsCompatibility(bil_potts));

    crf.getFeatureMat(img);

    MatrixXf Q;
    std::size_t found = image_file.find_last_of("/\\");
//    std::string image_name = image_file.substr(found+1);
    found = image_name.find_last_of(".");
    image_name = image_name.substr(0, found);
//    image_name = image_name + "_" + std::to_string(spc_std) + "_" + std::to_string(spc_potts) + "_" + std::to_string(bil_spcstd) + "_" + std::to_string(bil_colstd) + "_" + std::to_string(bil_potts);
    std::string path_to_subexp_results = path_to_results + "/" + dataset_name + "_" + method + "_" + std::to_string(spc_std) + "_" + std::to_string(spc_potts) + "_" + std::to_string(bil_spcstd) + "_" + std::to_string(bil_colstd) + "_" + std::to_string(bil_potts) + "/";
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
    } else if (method == "mf") {
        std::cout << "Starting mf inference " << std::endl;
        Q = crf.mf_inference(Q, output_path, dataset_name);
    } else if (method == "submod") {
        std::cout << "Starting submod inference " << std::endl;
        Q = unaries;
        Q = crf.submodular_inference(Q, size.width, size.height, output_path, dataset_name);
    } else if (method == "submod_rhst") {
        std::cout << "Starting submod tree inference " << std::endl;
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
     
    end = std::chrono::high_resolution_clock::now();
    timing = std::chrono::duration_cast<std::chrono::duration<double>>(end-start).count();
   save_map(Q, size, output_path, dataset_name);

}

int main(int argc, char *argv[])
{
    if (argc<4) {
        std::cout << "./generate split dataset method results_path spc_std spc_potts bil_spcstd bil_colstd bil_potts" << '\n';
        std::cout << "Example: ./generate Validation Pascal2010 method /data/MSRC/results/train/ 3 38 40 5 50" << '\n';
        return 1;
    }

    std::string dataset_split = argv[1];
    std::string dataset_name  = argv[2];
    std::string method = argv[3];
    std::string path_to_results = argv[4];

    std::cout << "Results path = " << path_to_results << std::endl;
//    float spc_std = 3.535267;
//    float spc_potts = 2.247081;
//    float bil_spcstd = 31.232626;
//    float bil_colstd = 7.949970;
//    float bil_potts = 1.699011;

    float spc_std = 0;
    float spc_potts = 0;
    float bil_spcstd = 0;
    float bil_colstd = 0;
    float bil_potts = 0;

 
   spc_std = std::stof(argv[5]);
   spc_potts = std::stof(argv[6]);
   bil_spcstd = std::stof(argv[7]);
   bil_colstd = std::stof(argv[8]);
   bil_potts = std::stof(argv[9]);

   std::cout << "#COMMAND: " << argv[0] << " " << dataset_split << " " << dataset_name << " " << method << " " << path_to_results << " " << spc_std << " " << spc_potts << " " << bil_spcstd << " " << bil_colstd << " " << bil_potts << std::endl;


    make_dir(path_to_results);

    Dataset ds = get_dataset_by_name(dataset_name);
    std::vector<std::string> test_images = ds.get_all_split_files(dataset_split);
    omp_set_num_threads(8);
	
    LP_inf_params lp_params;

//    #pragma omp parallel for
    for(int i=0; i< test_images.size(); ++i){
        image_inference(ds, method, path_to_results, test_images[i], spc_std, spc_potts,
                        bil_spcstd, bil_colstd, bil_potts, lp_params);
    }
}
