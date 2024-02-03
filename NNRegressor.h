#ifndef NN_NNREGRESSOR_H
#define NN_NNREGRESSOR_H
#include <array>
#include "Matrix.h"
#include "ActivationFunctions.h"

//double sigmoid(double x);
//double sigmoidDerivative(double x);

class NNRegressor {
public:
    using ActivationFunctionPointer = double (*)(double);
    NNRegressor(int hidden_layers, int depth, ActivationFunctionPointer activation_function = sigmoid, ActivationFunctionPointer activation_function_deriv = sigmoidDerivative);
    void init(std::vector<double>& in, std::vector<double>& out);
    void feed_forward();
    void back_propagate();
    void updateWeights(double learningRate);
    Matrix getOutput();
    void checkStatus();
    void updateLoss();

private:
    int num_hidden_layers;
    int depth;
    double currentLoss;
    ActivationFunctionPointer activationFunction;
    ActivationFunctionPointer activationFunctionDerivative;
    Matrix targetValues;
    Matrix inputValues;
    Matrix outputValues;
    Matrix inputWeights;
    Matrix outputWeights;
    Matrix outputWeightPartials;
    Matrix inputWeightPartials;
    Matrix outputError;
    std::vector<Matrix> hiddenLayerInputs;
    std::vector<Matrix> hiddenLayerOutputs;
    std::vector<Matrix> hiddenLayerErrors;
    std::vector<Matrix> hiddenLayerWeights;
    std::vector<Matrix> hiddenLayerPartials;
    std::vector<Matrix> hiddenLayerDerivatives;

};


#endif //NN_NNREGRESSOR_H
