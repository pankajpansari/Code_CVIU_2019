#ifndef CONVEX_EXAMPLE2_H
#define CONVEX_EXAMPLE2_H
#include <vector>

//struct FSize;
typedef struct FSize FSize;

float getFuncVal(std::vector<float> s, FSize f);
std::vector<float> getGradient(std::vector<float> s, FSize f);
std::vector<float> getConditionalGradient(std::vector<float> grad, FSize f, std::vector<std::vector <float>> unary);
std::vector<float> getInitialPoint(FSize f, std::vector<std::vector <float>> unary);

#endif
