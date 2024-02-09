#include <iostream>
#include "Matrix.h"
#include <algorithm>
#include "NNRegressor.h"
#include "NNClassifier.h"
#include <vector>
#include <fstream>
#include "ActivationFunctions.h"
#include "MnistLoader.h"
#include "FFClassifier.h"

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

void singleSample() {
    int numTrainingImages = 1000;
    int numTestImages = 10;
    int numEpocs = 5;
    int x = 7;

    MnistLoader mnistLoader(numTrainingImages, numTestImages);
    std::vector<double> targetValues(10);
    setTargetVector(targetValues, mnistLoader.trainingLabels[x]);

    auto network = std::make_unique<NNClassifier>(3,10);
    network->init(mnistLoader.trainingImages[x], targetValues);

    for (int i = 0; i < 1000; ++i) {
        network->feed_forward();
        network->back_propagate();

        network->printInputs();
        network->printOuputs();
        network->printTargets();
        network->printErrors();

        network->updateWeights(0.01);
    }
    network->feed_forward();
    network->getOutput().print();
    std::cout <<"Network trained \n";
    std::cout << "target: " << mnistLoader.trainingLabels[x];
}

void runMNIST(){
    int numTrainingImages = 1000;
    int numTestImages = 100;
    int numEpocs = 5;

    MnistLoader mnistLoader(numTrainingImages, numTestImages);
    std::vector<double> targetValues(10);


    auto network = std::make_unique<NNClassifier>(1,30);
    network->init(mnistLoader.trainingImages[0], targetValues);

    for (int k = 0; k < numEpocs; ++k) {
        for (int i = 0; i < numTrainingImages; ++i) {
            setTargetVector(targetValues, mnistLoader.trainingLabels[i]);
            network->loadDataPoint(mnistLoader.trainingImages[i], targetValues);
            network->feed_forward();
//            network->getOutput().print();
            network->back_propagate();
            network->updateWeights(0.1);
        }
        std::cout << "Epoc: " << k + 1 << "done\n";
    }
    std::cout <<"Network trained \n";

    int correctCount = 0;
    for (int i = 0; i < numTestImages; ++i) {
        setTargetVector(targetValues, mnistLoader.testLabels[i]);
        network->loadDataPoint(mnistLoader.testImages[i], targetValues);
        network->feed_forward();
        network->getOutput().print();
        std::cout << "target: "<< mnistLoader.testLabels[i] << "\n";
        //check the predicted value
        int highestIndex = 0;
        for (int j = 0; j < 10; ++j) {
            if (network->getOutput().getval(0, j) > network->getOutput().getval(0, highestIndex)) {
                highestIndex = j;
            }
        }
        if (highestIndex == mnistLoader.testLabels[i]) {
            correctCount += 1;
        }
    }
    std::cout << "We got " << correctCount << " out of " << numTestImages << " correct\n";
}

void batchProccess() {
    int numBatches = 10;
    int batchSize = 100;
    int numTrainingImages = numBatches*batchSize;
    int numTestImages = 100;
    int numEpocs = 5;

    MnistLoader mnistLoader(numTrainingImages, numTestImages);
    std::vector<double> targetValues(10);

    auto network = std::make_unique<NNClassifier>(1,200);
    network->init(mnistLoader.trainingImages[0], targetValues);

    for (int i = 0; i < numEpocs; ++i) {
        for (int j = 0; j < numBatches; ++j) {
            for (int k = 0; k < batchSize; ++k) {
                setTargetVector(targetValues, mnistLoader.trainingLabels[batchSize*j + k]);
                network->loadDataPoint(mnistLoader.trainingImages[batchSize*j + k], targetValues);
                network->feed_forward();
                network->back_propagate();
                network->batchSingleUpdate();
            }
            network->batchFullUpdate(0.05, batchSize);
        }
        std::cout << "Epoc: " << i + 1 << "done\n";
    }
    std::cout <<"Network trained \n";

    int correctCount = 0;
    for (int i = 0; i < numTestImages; ++i) {
        setTargetVector(targetValues, mnistLoader.testLabels[i]);
        network->loadDataPoint(mnistLoader.testImages[i], targetValues);
        network->feed_forward();
        network->getOutput().print();
        std::cout << "target: "<< mnistLoader.testLabels[i] << "\n";
        //check the predicted value
        int highestIndex = 0;
        for (int j = 0; j < 10; ++j) {
            if (network->getOutput().getval(0, j) > network->getOutput().getval(0, highestIndex)) {
                highestIndex = j;
            }
        }
        if (highestIndex == mnistLoader.testLabels[i]) {
            correctCount += 1;
        }
    }
    std::cout << "We got " << correctCount << " out of " << numTestImages << " correct\n";
}

void binaryTest() {
    int numTrainingImages = 1000;
    int numTestImages = 10;
    int numEpocs = 5;
    int x = 4;

    MnistLoader mnistLoader(numTrainingImages, numTestImages);
    for (int i = 0; i < mnistLoader.trainingImages.size(); ++i) {
        if (mnistLoader.trainingLabels[i] != 0 || mnistLoader.trainingLabels[i] != 1) {
            mnistLoader.trainingLabels.erase(mnistLoader.trainingLabels.begin()+i);
            mnistLoader.trainingImages.erase(mnistLoader.trainingImages.begin()+i);
        }
    }
    for (int i = 0; i < mnistLoader.testLabels.size(); ++i) {
        if (mnistLoader.testLabels[i] != 0 || mnistLoader.testLabels[i] != 1) {
            mnistLoader.testLabels.erase(mnistLoader.testLabels.begin()+i);
            mnistLoader.testImages.erase(mnistLoader.testImages.begin()+i);
        }
    }
    std::vector<double> targetValues(2);
    setTargetVector(targetValues, mnistLoader.trainingLabels[0]);

    auto network = std::make_unique<NNClassifier>(2,50);
    network->init(mnistLoader.trainingImages[0], targetValues);

    for (int i = 0; i < 1000; ++i) {
        setTargetVector(targetValues, mnistLoader.trainingLabels[i]);
        network->loadDataPoint(mnistLoader.trainingImages[i], targetValues);
        network->feed_forward();
        network->back_propagate();
//        network->printInputs();
//        network->printOuputs();
//        network->printTargets();
//        network->printErrors();
        network->updateWeights(0.01);
    }
    network->feed_forward();
    network->getOutput().print();
    std::cout <<"Network trained \n";
    std::cout << "target: " << mnistLoader.trainingLabels[x];
}
int main() {
//    runTests();
    int trainingImages = 1000;
    int testImages = 100;
    MnistLoader mnistLoader(trainingImages, testImages);
    FFClassifier network({784,30,10}, 10);
//    for (int i = 0; i < 30; i++) {
//        network.train(mnistLoader);
//    }
    network.train(mnistLoader);
    network.test(mnistLoader);
    return 0;
}
