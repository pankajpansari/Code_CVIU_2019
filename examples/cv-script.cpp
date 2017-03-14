#include <fstream>
#include <vector>
#include <string>
#include "file_storage.hpp"
#include "densecrf.h"

void image_inference(Dataset dataset, std::string method, std::string path_to_results,
                     std::string image_name, float spc_std, float spc_potts, float bil_spcstd, float bil_colstd, float bil_potts, LP_inf_params & lp_params)
{

    std::string image_path = dataset.get_image_path(image_name);
    std::string unaries_path = dataset.get_unaries_path(image_name);
    std::string dataset_name = dataset.name;

    img_size size = {-1, -1};
    // Load the unaries potentials for our image.
    MatrixXf unaries = load_unary(unaries_path, size);
    unsigned char * img = load_image(image_path, size);

    DenseCRF2D crf(size.width, size.height, unaries.rows());
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));
    crf.addPairwiseBilateral(bil_spcstd, bil_spcstd,
                             bil_colstd, bil_colstd, bil_colstd,
                             img, new PottsCompatibility(bil_potts));


    std::cout << "Method = " << method << std::endl;
    MatrixXf Q;
    {
        std::string path_to_subexp_results = path_to_results + "/" + method + "/";
        std::string output_path = get_output_path(path_to_subexp_results, image_name);
        if (not file_exist(output_path)) {
            clock_t start, end;
            double timing;
            std::cout << image_path << std::endl;
            start = clock();
            Q = crf.unary_init();
            if (method == "mf5") {
                Q = crf.mf_inference(Q, 5);
            } else if (method == "mf") {
                Q = crf.mf_inference(Q);
            } else if (method == "lrqp") {
                Q = crf.qp_inference(Q);
            } else if (method == "qpcccp") {
                Q = crf.qp_inference(Q);
                Q = crf.qp_cccp_inference(Q);
            } else if (method == "fixedDC-CCV"){
                Q = crf.qp_inference(Q);
                Q = crf.concave_qp_cccp_inference(Q);
            } else if (method == "sg_lp"){
                Q = crf.qp_inference(Q);
                Q = crf.concave_qp_cccp_inference(Q);
                Q = crf.lp_inference(Q, false);
            } else if (method == "cg_lp"){
                Q = crf.qp_inference(Q);
                Q = crf.concave_qp_cccp_inference(Q);
                Q = crf.lp_inference(Q, true);
            } else if (method == "prox-lp") {    // standard prox_lp
		Q = crf.qp_inference(Q);
		Q = crf.concave_qp_cccp_inference(Q);
		Q = crf.lp_inference_prox(Q, lp_params);    
	    } else if (method == "unary"){
                (void)0;
            } else{
                std::cout << "Unrecognised method.\n Proper error handling would do something but I won't." << '\n';
            }


            make_dir(path_to_subexp_results);
            end = clock();
            timing = (double(end-start)/CLOCKS_PER_SEC);
            double final_energy = crf.compute_energy(Q);
            double discretized_energy = crf.assignment_energy(crf.currentMap(Q));
            save_map(Q, size, output_path, dataset_name);
            std::string txt_output = output_path;
            txt_output.replace(txt_output.end()-3, txt_output.end(),"txt");
            std::ofstream txt_file(txt_output.c_str());
            txt_file << timing << '\t' << final_energy << '\t' << discretized_energy << std::endl;
            txt_file.close();
        }
    }
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

    float spc_std = 3.535267;
    float spc_potts = 2.247081;
    float bil_spcstd = 31.232626;
    float bil_colstd = 7.949970;
    float bil_potts = 1.699011;

     if (argc < 10) { 
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
        spc_std = std::stof(argv[5]);
        spc_potts = std::stof(argv[6]);
        bil_spcstd = std::stof(argv[7]);
        bil_colstd = std::stof(argv[8]);
        bil_potts = std::stof(argv[9]);
    }

    make_dir(path_to_results);

    std::cout << "Hello, world!" << std::endl;
    Dataset ds = get_dataset_by_name(dataset_name);
    std::vector<std::string> test_images = ds.get_all_split_files(dataset_split);
    omp_set_num_threads(8);
    std::cout << "Dataset size = " << test_images.size() << std::endl;
	
    LP_inf_params lp_params;

#pragma omp parallel for
    for(int i=0; i< test_images.size(); ++i){
        image_inference(ds, method, path_to_results, test_images[i], spc_std, spc_potts,
                        bil_spcstd, bil_colstd, bil_potts, lp_params);
    }
}
