#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include "ActivationFunctions.h"
#include "MnistLoader.h"

extern void runTests();

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

void setTargetVector(std::vector<double>& vec,double target) {
    for (auto &x: vec) {
        x = 0;
    }
    vec[static_cast<int>(target)] = 1;
}


std::vector<double> one_hot(std::vector<double>& targets) {
    std::vector<double> output;
    for (auto &target: targets) {
        std::vector<double> vector(10);
        for (auto &x: vector) {
            x = 0;
        }
        vector[static_cast<int>(target)] = 1.0;
        output.insert(output.end(), vector.begin(), vector.end());
    }
    return  output;
}



int main() {
////    runTests();
    int trainingImages = 10000;
    int testImages = 1000;
    MnistLoader mnistLoader(trainingImages, testImages);
    return 0;
}
