include_directories(../external/ProbImage ../external/ )

find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ${OpenMP_CXX_FLAGS}")
endif()

add_library(color_to_label color_to_label.cpp)

add_library(file_storage file_storage.cpp)
target_link_libraries(file_storage probimage color_to_label ${OpenCV_LIBS})

# Executables
# inference.cpp
add_executable(inference inference.cpp)
target_link_libraries(inference densecrf file_storage densecrf_utils)

#add_executable(gradientDescent gradientDescentTest.cpp)
#target_link_libraries(gradientDescent densecrf file_storage densecrf_utils)

# cv-script.cpp 
#add_executable(msrc_inference cv-script.cpp)
#target_link_libraries(msrc_inference densecrf file_storage densecrf_utils)

# toyExample.cpp
#add_executable(toyExample toyExample.cpp)
#target_link_libraries(toyExample densecrf file_storage densecrf_utils)

# trial.cpp
add_executable(trial trial.cpp)
target_link_libraries(trial densecrf file_storage densecrf_utils)
