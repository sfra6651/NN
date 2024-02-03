#include <iostream>
#include "Matrix.h"
#include <algorithm>
#include "NNRegressor.h"
#include "NNClassifier.h"
#include <vector>
#include <fstream>
#include "ActivationFunctions.h"

extern void runTests();



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
//    runTests();

    std::vector<double> out {};
    std::vector<double> input {};
    textToVector(input, std::string("/Users/shaun/Dev/NN/input.txt"));
    textToVector(out, std::string("/Users/shaun/Dev/NN/output.txt"));

    auto network = std::make_unique<NNClassifier>(3,5);

//    Network network(3,10);
    network->init(input, out);

    for (int i = 0; i < 1000; ++i) {
        network->feed_forward();
        network->back_propagate();
        network->updateWeights(0.01);
    }
    network->feed_forward();
    std::cout << "FINAL PREDICTION: \n";
    network->getOutput().print();

//    network->checkStatus();

    return 0;
}
