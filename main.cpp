#include <iostream>
#include "Matrix.h"
#include "Neuron.h"
#include <algorithm>
#include "Network.h"
#include <vector>
#include <fstream>


//bool activation_f(double sum) {
//    return static_cast<bool>(std::max(0.0, sum));
//}

void textToVector(std::vector<double>& vector, std::string string) {
    std::ifstream file{string};
    if (!file)
    {
        // Print an error and exit
        std::cerr << "Uh oh, Sample.txt could not be opened for reading!\n";
    }
    while(file){
        std::string strInput;
        file >> strInput;
        if(std::isdigit(strInput[0])) {
            vector.push_back(std::stod(strInput));
        }
    }
}


int main() {
    std::vector<double> va {{3,4,2}};
    std::vector<double> vb {{13,9,7,15,8,7,4,6,6,4,0,3}};
    Matrix matrixA(1,3,va);
    Matrix matrixB(3,4,vb);
    Matrix product(1,4);

//    matrixA.print();
//    std::cout << "\n";
//
//    matrixB.print();
//    matrix_multiply(matrixA , matrixB , product);
//    std::cout <<"\nafter Mutiplication " << "\n";
//    product.print();

    std::vector<double> out {};
    std::vector<double> input {};
    textToVector(input, std::string("/Users/shaun/Dev/NN/input.txt"));
    textToVector(out, std::string("/Users/shaun/Dev/NN/output.txt"));

    Network network(3,10);

    network.init(input, out);
    network.feed_forward();

    return 0;
}
