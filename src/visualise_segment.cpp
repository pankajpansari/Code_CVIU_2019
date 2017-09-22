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
   
int main(int argc, char* argv[]) 
{
       img_size size = {-1, -1};
       unary_file = 
       unaries = load_unary_rescaled(unary_file, size, 1);
       //read marginals

       save_map(marginals, size, "./img.png", "MSRC");
}

