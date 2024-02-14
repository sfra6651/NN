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
    vec[static_cast<int>(target)] = 1;
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

std::pair<std::vector<Matrix>, std::vector<Matrix>> init(std::vector<int> sizes) {
    std::vector<Matrix> biases;
    std::vector<Matrix> weights;
    for (int i = 1; i < sizes.size(); ++i) {
        biases.push_back(Matrix(sizes[i],1, true));
        weights.push_back(Matrix(sizes[i],sizes[i-1], true));
    }

    return {weights, biases};
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> feed_forward(
        std::pair<std::vector<Matrix>, std::vector<Matrix>>& init, Matrix& X)
{
    std::vector<Matrix> inputs;
    std::vector<Matrix> activations;
    activations.push_back(X);
    for (int i = 0; i < init.first.size(); ++i) {
        inputs.push_back(init.first[i].dot(activations[activations.size()-1]));
        Matrix temp = Matrix(inputs[0].getrows(), inputs[0].getcols());
        inputs[0].mapto(sigmoid, temp);
        activations.push_back(temp);
    }
    return {inputs, activations};
}

std::pair<std::vector<Matrix>, std::vector<Matrix>> back_prop(
        std::pair<std::vector<Matrix>, std::vector<Matrix>>& inputs_activations,
        std::vector<Matrix>& weights,
        Matrix& Y
        )
{
    int m = Y.getcols();
    std::vector<Matrix> dW;
    std::vector<Matrix> dB;
    std::vector<Matrix> dZ;
    Matrix dz1 = inputs_activations.second.back() - Y;
    dZ.push_back(dz1);
    for (int i = weights.size()-1; i > -1; --i) {

    }

}



int main() {
////    runTests();
    int trainingImages = 10000;
    int testImages = 1000;
    MnistLoader mnistLoader(trainingImages, testImages);
    std::vector<double> temp;
    for (int i = 0; i < mnistLoader.trainingImages.size(); ++i) {
        temp.insert(temp.end(), mnistLoader.trainingImages[i].begin(), mnistLoader.trainingImages[i].end());
    }
    Matrix X_train = Matrix(mnistLoader.trainingImages.size(), 784, temp);

    std::vector<double> one_hoted = one_hot(mnistLoader.trainingLabels);
    Matrix Y_train = Matrix(mnistLoader.trainingLabels.size(), 10, one_hoted);

    X_train.transposeInplace();
    Y_train.transposeInplace();

    std::pair<std::vector<Matrix>, std::vector<Matrix>> weights_biases;
    std::pair<std::vector<Matrix>, std::vector<Matrix>> inputs_activations;

    weights_biases = init({784,30,10});
    inputs_activations = feed_forward(weights_biases, X_train);




//    FFClassifier network({784,30,10}, 10);
//    for (int i = 0; i < 30; i++) {
//        network.train(mnistLoader);
//        network.test(mnistLoader);
//    }
//    network.train(mnistLoader);
//    network.test(mnistLoader);

    return 0;
}
