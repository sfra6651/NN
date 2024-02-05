#include <iostream>
#include "Matrix.h"
#include <algorithm>
#include "NNRegressor.h"
#include "NNClassifier.h"
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
    vec[target] = 1;
}

void runMNIST(){
    int numTrainingImages = 1000;
    int numTestImages = 100;
    int numEpocs = 5;

    MnistLoader mnistLoader(numTrainingImages, numTestImages);
    std::vector<double> targetValues(10);


    auto network = std::make_unique<NNClassifier>(1,512);
    network->init(mnistLoader.trainingImages[0].pixels, targetValues);

    for (int k = 0; k < numEpocs; ++k) {
        for (int i = 0; i < numTrainingImages; ++i) {
            setTargetVector(targetValues, mnistLoader.trainingLabels[i]);
            network->loadDataPoint(mnistLoader.trainingImages[i].pixels, targetValues);
            network->feed_forward();
//            network->getOutput().print();
            network->back_propagate();
            network->updateWeights(0.01);
        }
        std::cout << "Epoc: " << k + 1 << "done\n";
    }
    std::cout <<"Network trained \n";

    int correctCount = 0;
    for (int i = 0; i < numTestImages; ++i) {
        setTargetVector(targetValues, mnistLoader.testLabels[i]);
        network->loadDataPoint(mnistLoader.testImages[i].pixels, targetValues);
        network->feed_forward();
        network->getOutput().print();
        //check the predicted value
        int highestIndex = 0;
        for (int j = 1; j < 10; ++j){
            if (network->getOutput().getval(0, j) > highestIndex) {
                highestIndex = j;
            }
        }
        if (highestIndex == mnistLoader.testLabels[i]) {
            correctCount += 1;
        }
    }
    std::cout << "We got " << correctCount << " out of " << numTestImages << " correct\n";
}


int main() {
    runTests();
    runMNIST();

//    int numTrainingImages = 10;
//    int numTestImages = 10;
//    int numEpocs = 5;
//    int x = 4;

//    MnistLoader mnistLoader(numTrainingImages, numTestImages);
//    std::vector<double> targetValues(10);
//    setTargetVector(targetValues, mnistLoader.trainingLabels[x]);
//
//    auto network = std::make_unique<NNClassifier>(3,10);
//    network->init(mnistLoader.trainingImages[x].pixels, targetValues);
//
//    for (int i = 0; i < 100; ++i) {
//        network->feed_forward();
//        network->back_propagate();
//        network->updateWeights(0.1);
//    }
//    std::cout <<"Network trained \n";
//    std::cout << "target: " << mnistLoader.trainingLabels[x];



    return 0;
}
