#ifndef FRANKWOLFE_H
#define FRANKWOLFE_H

#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

typedef struct FSize FSize;

std::vector<float> frankWolfeAlgo(FSize f, std::vector<std::vector <float>> unary);

#endif
