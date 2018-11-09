include_directories( ../external/ )
include_directories( ../examples/ )

# Libraries
add_library(densecrf_utils densecrf_utils.cpp)
add_library(tree_utils tree_utils.cpp)
add_library(sparsecrf sparsecrf.cpp)
#add_library(densecrf densecrf.cpp)
add_library(densecrf permutohedral.cpp unary.cpp pairwise.cpp densecrf.cpp submodular.cpp submodular_rhst.cpp matrix_prod.cpp labelcompatibility.cpp )

#target_link_libraries(densecrf densecrf_utils file_storage tree_utils)
#target_link_libraries(sparsecrf densecrf_utils file_storage tree_utils)

