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
        
//    MatrixXf unaries = load_unary_rescaled(unary_file, size, imskip);
   MatrixXf unaries = load_unary(unary_file, size);
 //   unsigned char * img = load_rescaled_image(image_file, size, imskip);
   unsigned char * img = load_image(image_file, size);
    
    // create densecrf object
    DenseCRF2D crf(size.width, size.height, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
    crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,
                             bil_colstd, bil_colstd, bil_colstd,
                             img, new PottsCompatibility(bil_potts));

    crf.getFeatureMat(img);

    MatrixXf Q;
    std::cout << image_file << std::endl;
//    std::size_t found = image_file.find_last_of("/\\");
//    std::string image_name = image_file.substr(found+1);
//    found = image_name.find_last_of(".");
//    image_name = image_name.substr(0, found);
    std::string path_to_subexp_results = path_to_results + "/" + method + "/";
    std::string output_path = get_output_path(path_to_subexp_results, image_name);
    make_dir(path_to_subexp_results);

    typedef std::chrono::high_resolution_clock::time_point htime;
    htime start, end;
    double timing;
    std::vector<int> pixel_ids;
    start = std::chrono::high_resolution_clock::now();

    Q = unaries;

//    double initial_discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));
//   std::cout << "Initial energy = " << initial_discretized_energy << std::endl;
 

    if (method == "mf5") {
        Q = crf.mf_inference(Q, 5);
    } else if (method == "mf") {
        Q = crf.mf_inference(Q);
    } else if (method == "submod") {
        Q = crf.submodular_inference(Q, size.width, size.height, output_path);
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
    double final_energy = crf.compute_energy_true(Q);
    double discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));
    save_map(Q, size, output_path, dataset_name);
    if (!pixel_ids.empty()) save_less_confident_pixels(Q, pixel_ids, size, output_path, dataset_name);
    std::string txt_output = output_path;
    txt_output.replace(txt_output.end()-3, txt_output.end(),"txt");
    std::ofstream txt_file(txt_output);
    txt_file << timing << '\t' << final_energy << '\t' << discretized_energy << std::endl;
    std::cout << "#" << method << ": " << timing << '\t' << final_energy << '\t' << discretized_energy << std::endl;
    txt_file.close();
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
