#include <iostream>
#include "Matrix.h"
#include "Neuron.h"
#include <algorithm>
#include "Network.h"
#include <vector>
#include <fstream>

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
    runTests();

    std::vector<double> out {};
    std::vector<double> input {};
    textToVector(input, std::string("/Users/shaun/Dev/NN/input.txt"));
    textToVector(out, std::string("/Users/shaun/Dev/NN/output.txt"));

    auto network = std::make_unique<Network>(3,10);

//    Network network(3,10);
    network->init(input, out);
    network->feed_forward();
    network->back_propagate();

    return 0;
}
