#ifndef NN_NETWORK_H
#define NN_NETWORK_H
#include <array>
#include "Matrix.h"
#include "Neuron.h"

double relU(double sum);
double relUDerivative(double x);

class Network {
public:
    using ActivationFunctionPointer = double (*)(double);
    Network(int hidden_layers, int depth, ActivationFunctionPointer activation_function = relU);
    void init(std::vector<double>& in, std::vector<double>& out);
    void feed_forward();
    void back_propagate();
    void updateLoss();

private:
    int num_hidden_layers;
    int depth;
    double currentLoss;
    ActivationFunctionPointer activationFunction;
    std::vector<std::vector<Neuron>> hiddenLayers;
    Matrix targetValues;
    Matrix inputValues;
    Matrix outputValues;
    Matrix inputWeights;
    Matrix outputWeights;
    Matrix outputWeightDeltas;
    Matrix inputWeightDeltas;
    Matrix outputError;
    std::vector<Matrix> hiddenLayerInputs;
    std::vector<Matrix> hiddenLayerOutputs;
    std::vector<Matrix> hiddenLayerWeights;
    std::vector<Matrix> hiddenLayerErrors;
    std::vector<Matrix> hiddenLayerDerivatives;

};


#endif //NN_NETWORK_H
