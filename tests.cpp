#include "Matrix.h"
#include "NNRegressor.h"
#include <vector>
#include <iostream>
#include "MnistLoader.h"

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

    std::cout << "TEST: Matrix addition verified successfully." << std::endl;

}

void matrixSubtract(){
    std::vector<double> va{{2 ,3, 2, 0, 2, 3, 1, 1}};
    std::vector<double> vb{{1 ,1, 1, 1, 3, 1, 1, 0}};
    std::vector<double> output{{1,2,1,-1,-1,2,0,1}};

    Matrix A(1,va.size(), va);
    Matrix B(1,vb.size(), vb);
    Matrix OUTPUT(1, output.size(), output);

    Matrix x{A-B};
    if (!(x==OUTPUT)) {
        throw std::runtime_error("Matrix subtraction result does not return the correct matrix");
    }

    A.subtract(B);

    std::vector<double> Avec = A.getVector();
    std::vector<double> OUTPUTvec = OUTPUT.getVector();

    for (size_t i = 0; i < Avec.size(); ++i) {
        if (Avec[i] != OUTPUTvec[i]) {
            // Handle the error - either throw an exception, return an error code, or some other error handling mechanism
            throw std::runtime_error("Matrix subtraction result does not match expected output at index " + std::to_string(i));
        }
    }

    std::cout << "TEST: Matrix subtraction verified successfully." << std::endl;

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

    std::cout << "TEST: Matrix equality verified successfully." << std::endl;
}

void scalar_multiply(){
    std::vector<double> va{{2 ,2, 2, 4}};
    std::vector<double> vb{{-1 ,-2, -3, -4}};
    std::vector<double> oa{{4 ,4, 4, 8}};
    std::vector<double> ob{{-2 ,-4, -6, -8}};

    Matrix A(1,va.size(), va);
    Matrix B(1,vb.size(), vb);
    Matrix C(1, oa.size(), oa);
    Matrix D(1, ob.size(), ob);

    A.scalarMultiply(2.0);
    B.scalarMultiply(2.0);

    if(!(A==C) || !(B==D)) {
        throw std::runtime_error("scalar multiply failed");
    } else {
        std::cout << "TEST: scalar multiply successful\n";
    }

}

void matrixFrom2Vectors(){
    std::vector<double> va{{2 ,3}};
    std::vector<double> vb{{2 ,3}};
    std::vector<double> real{{4,6,6,9}};
    Matrix A(va, vb);
    Matrix B(2,2,real);
    if(!(A==B)) {
        throw std::runtime_error("making matrix From 2 Vectors failed");
    }
    std::cout << "TEST: making matrix From 2 Vectors succeeded\n";
}

void matrixTranspose() {
    std::vector<double> va{{1,2,3,4,5,6,1,2,3}};
    std::vector<double> vb{{1,4,1,2,5,2,3,6,3}};
    Matrix A(3,3,va);
    Matrix B(3,3,vb);
//    A.print();
//    std::cout << "\n";
    A.transpose();
//    A.print();
    if (!(A==B)) {
        throw std::runtime_error("matrix transpose failed failed");
    }
    std::cout << "TEST: matrix transpose succeeded\n";
}

void elementWiseMultiply(){
    std::vector<double> va{{1,2,3,1,2,3}};
    std::vector<double> vb{{2,2,3,2,2,3}};
    Matrix A(2,3,va);
    Matrix B(2,3,vb);

    A.elementwiseMultiply(B);
//    A.print();
}

void MnistLoaderTest(){
    int image = 16;
    int numTrainingImages = 100;
    int numTestImages = 100;
    MnistLoader mnistLoader(numTrainingImages, numTestImages);
    std::cout << "Label: " << mnistLoader.testLabels[image] << "\n";
    for (int i = 0; i < 28; ++i) {
        for(int j = 0; j < 28; ++j) {
            if(mnistLoader.testImages[image][28*i + j] > 0) {
                if(mnistLoader.testImages[image][28*i + j] > 128) {
                    std::cout << "0";
                } else {
                    std::cout << "-";
                }
            } else {
                std::cout << " ";
            }
        }
        std::cout << "\n";
    }

}

void runTests(){
//    matrixAdd();
//    matrixSubtract();
//    matrixEquality();
//    elementWiseMultiply();
//    scalar_multiply();
//    matrixFrom2Vectors();
//    matrixTranspose();
    MnistLoaderTest();
}
