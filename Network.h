#ifndef NN_NETWORK_H
#define NN_NETWORK_H
#include <array>
#include "Matrix.h"
#include "Neuron.h"

double relU(double sum);

class Network {
public:
    using ActivationFunctionPointer = double (*)(double);
    Network(int hidden_layers, int depth, ActivationFunctionPointer activation_function = relU);
    void init(std::vector<double>& in, std::vector<double>& out);
    void feed_forward();
    void back_propagation();
    void getLoss();

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
    std::vector<Matrix> hiddenLayerInputs;
    std::vector<Matrix> hiddenLayerOutputs;
    std::vector<Matrix> hiddenLayerWeights;

};


#endif //NN_NETWORK_H
