#!/bin/bash
./inference ../../data/stereo/tsukuba_left.png ../../data/stereo/confFileStereo_tsukuba.txt unary /home/pankaj/SubmodularInference/data/working/24_10_2017 Stereo_special 1.640000 12.990000 50.840000 1.510000 1.100000
./inference ../../data/stereo/teddy_left.png ../../data/stereo/confFileStereo_teddy.txt unary /home/pankaj/SubmodularInference/data/working/24_10_2017 Stereo_special 1.640000 12.990000 50.840000 1.510000 1.100000

./inference ../../data/stereo/cones_left.png ../../data/stereo/confFileStereo_cones.txt unary /home/pankaj/SubmodularInference/data/working/24_10_2017 Stereo_special 1.640000 12.990000 50.840000 1.510000 1.100000
./inference ../../data/stereo/venus_left.png ../../data/stereo/confFileStereo_venus.txt unary /home/pankaj/SubmodularInference/data/working/24_10_2017 Stereo_special 1.640000 12.990000 50.840000 1.510000 1.100000
#./inference ../../data/stereo/tsukuba_left.png ../../data/stereo/confFileStereo_tsukuba.txt submod /home/pankaj/SubmodularInference/data/working/24_10_2017 Stereo_special 1.640000 12.990000 50.840000 1.510000 1.100000
#./inference ../../data/stereo/teddy_left.png ../../data/stereo/confFileStereo_teddy submod /home/pankaj/SubmodularInference/data/working/23_10_2017 Stereo_special 1.640000 12.990000 50.840000 1.510000 1.100000
#
#./inference ../../data/stereo/cones_left.png ../../data/stereo/confFileStereo_cones.txt submod /home/pankaj/SubmodularInference/data/working/23_10_2017 Stereo_special 1.640000 12.990000 50.840000 1.510000 1.100000
#./inference ../../data/stereo/venus_left.png ../../data/stereo/confFileStereo_venus.txt submod /home/pankaj/SubmodularInference/data/working/23_10_2017 Stereo_special 1.640000 12.990000 50.840000 1.510000 1.100000
