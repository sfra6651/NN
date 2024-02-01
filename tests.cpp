#include "Matrix.h"
#include "Network.h"
#include <vector>

void matrixAdd(Matrix& a, Matrix* b){

}

void matrixTest() {
    std::vector<double> va{{1 ,2, 3, 4, 5, 6, 7, 8}};
    std::vector<double> vb{{1 ,2, 3, 4, 5, 6, 7, 8}};
    std::vector<double> add_output{{2,4,6,8,10,12,14,16}};

    Matrix A(1,va.size(), va);
    Matrix B(1,vb.size(), vb);
    static_assert(A.add(B) == std::vector<double>{2,4,6,8,10,12,14,16});
}

void runTests(){

}
