//template
//compile & link - g++ trial.cpp
//execute	./a.out
#include <iostream>
#include <vector>
#include <algorithm>

#include <Eigen>
using namespace Eigen;
using namespace std;

int main()
{
    
    std::vector<int> a = {2, 1, 4, 3};

    std::vector<int> y(a.size());
    for(int i = 0; i < y.size(); i++)
        y[i] = i;
 
    auto comparator = [&a](int p, int q){ return a[p] > a[q]; };
    sort(y.begin(), y.end(), comparator);

    for(int i = 0; i < y.size(); i++)
        std::cout << y[i] << std::endl;

    std::cout << a.size() << std::endl;
    if(std::find(a.begin(), a.end(), 5) != a.end())
        std::cout << "present" << std::endl;
    else
        std::cout << "not present" << std::endl;
}
