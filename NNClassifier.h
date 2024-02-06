#ifndef NN_NNCLASSIFIER_H
#define NN_NNCLASSIFIER_H

#include "Matrix.h"
#include "ActivationFunctions.h"
#include "ActivationFunctions.h"

class NNClassifier {
public:
    using ActivationFunctionPointer = double (*)(double);
    NNClassifier(int hidden_layers, int depth, ActivationFunctionPointer activation_function = sigmoid, ActivationFunctionPointer activation_function_deriv = sigmoidDerivative);
    void setOuptupLayerFunction(ActivationFunctionPointer f = sigmoid, ActivationFunctionPointer fd = sigmoidDerivative);
    void init(std::vector<double>& in, std::vector<double>& out);
    void feed_forward();
    void back_propagate(double numSamples = 1.0);
    void updateWeights(double learningRate);
    void batchSingleUpdate();
    void batchFullUpdate(double learnginRate, double batchSize);
    void loadDataPoint(std::vector<double>& pointInput, std::vector<double>& target);
    Matrix getOutput();
    void printErrors();
    void printOuputs();
    void printInputs();
    void printTargets();
    void checkStatus();
    void updateLoss();

private:
    int num_hidden_layers;
    int depth;
    double currentLoss;
    ActivationFunctionPointer activationFunction;
    ActivationFunctionPointer activationFunctionDerivative;
    ActivationFunctionPointer outputLayerActivationF;
    ActivationFunctionPointer outputLayerActivationFDerivative;
    Matrix targetValues;
    Matrix inputValues;
    Matrix outputLayerInputs;
    Matrix outputLayerOuputs;
    Matrix outputLayerDerivatives;
    Matrix inputWeights;
    Matrix outputWeights;
    Matrix outputWeightPartials;
    Matrix inputWeightPartials;
    Matrix outputLayerError;
    Matrix outputLayerBias;
    std::vector<Matrix> hiddenLayerInputs;
    std::vector<Matrix> hiddenLayerOutputs;
    std::vector<Matrix> hiddenLayerErrors;
    std::vector<Matrix> hiddenLayerWeights;
    std::vector<Matrix> hiddenLayerPartials;
    std::vector<Matrix> hiddenLayerDerivatives;
    std::vector<Matrix> hiddenLayerBiases;

    Matrix inputBatchWeights;
    Matrix outputBatchWeighs;
    std::vector<Matrix> hiddenLayerBatchWeights;

};


#endif //NN_NNCLASSIFIER_H
