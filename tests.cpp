#include "Matrix.h"
#include "Network.h"
#include <vector>
#include <iostream>

void matrixAdd(){
    std::vector<double> va{{1 ,2, 3, 4, 5, 6, 7, 8}};
    std::vector<double> vb{{1 ,2, 3, 4, 5, 6, 7, 8}};
    std::vector<double> output{{2,4,6,8,10,12,14,16}};

    Matrix A(1,va.size(), va);
    Matrix B(1,vb.size(), vb);
    Matrix OUTPUT(1, output.size(), output);

    A.add(B);

    std::vector<double> Avec = A.getVector();
    std::vector<double> OUTPUTvec = OUTPUT.getVector();

    for (size_t i = 0; i < Avec.size(); ++i) {
        if (Avec[i] != OUTPUTvec[i]) {
            // Handle the error - either throw an exception, return an error code, or some other error handling mechanism
            throw std::runtime_error("Matrix addition result does not match expected output at index " + std::to_string(i));
        }
    }

    std::cout << "Matrix addition verified successfully." << std::endl;

}

void matrixSubtract(){
    std::vector<double> va{{2 ,3, 2, 0, 2, 3, 1, 1}};
    std::vector<double> vb{{1 ,1, 1, 1, 3, 1, 1, 0}};
    std::vector<double> output{{1,2,1,-1,-1,2,0,1}};

    Matrix A(1,va.size(), va);
    Matrix B(1,vb.size(), vb);
    Matrix OUTPUT(1, output.size(), output);



    A.subtract(B);

    std::vector<double> Avec = A.getVector();
    std::vector<double> OUTPUTvec = OUTPUT.getVector();

    for (size_t i = 0; i < Avec.size(); ++i) {
        if (Avec[i] != OUTPUTvec[i]) {
            // Handle the error - either throw an exception, return an error code, or some other error handling mechanism
            throw std::runtime_error("Matrix subtraction result does not match expected output at index " + std::to_string(i));
        }
    }

    std::cout << "Matrix subtraction verified successfully." << std::endl;

}

void matrixEquality(){
    std::vector<double> va{{1 ,2, 2, 0, 2, 3, 1, 1}};
    std::vector<double> vb{{1 ,2, 2, 0, 2, 3, 1, 1}};
    std::vector<double> vc{{1,2,1,-1,-1,2,0,1}};
    std::vector<double> vd{{1 ,2, 2, 0, 2, 3, 1, 1}};

    Matrix A(1,va.size(), va);
    Matrix B(1,vb.size(), vb);
    Matrix C(1, vc.size(), vc);
    Matrix D(2, vd.size()/2, vd);

    if (!(A == B)) {
        throw std::runtime_error("Equality failed for equal matrices");
    }
    if (A == D) {
        throw std::runtime_error("Equality failed for matrices of different sizes with the same vector");
    }
    if (A == C) {
        throw std::runtime_error("Equality failed for unequal  matrices of same size");
    }

    std::cout << "Matrix equality verified successfully." << std::endl;
}

void runTests(){
//    matrixAdd();
//    matrixSubtract();
//    matrixEquality();
}
