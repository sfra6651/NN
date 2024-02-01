#include "Network.h"
#include "Neuron.h"
#include "iostream"

Network::Network(int hidden_layers, int depth, ActivationFunctionPointer activation_function)
    :num_hidden_layers{hidden_layers},
    depth{depth},
    activationFunction {activation_function}
{
}

void Network::init(std::vector<double> &input, std::vector<double> &output) {
//    std::cout << "input/out weights starting\n";
    inputValues = Matrix(1,input.size(), input);
    outputValues = Matrix(1,output.size(), output);

    inputWeights = Matrix(input.size(),depth, true);
    outputWeights = Matrix(depth,output.size(), true);

    targetValues = Matrix(1, output.size() ,output);
//    std::cout << "input/out weights assinged\n";

    for (int i = 0; i < num_hidden_layers - 1;  ++i) {
        hiddenLayerWeights.push_back(Matrix(depth, depth, true));
    }
    for (int i = 0; i < num_hidden_layers;  ++i) {
        hiddenLayerOutputs.push_back(Matrix(1,depth));
        hiddenLayerInputs.push_back(Matrix(1,depth));
    }
}

void Network::feed_forward() {
    matrix_multiply(inputValues, inputWeights, hiddenLayerInputs[0]);
    hiddenLayerInputs[0].mapto(activationFunction, hiddenLayerOutputs[0]);

    for(int i = 0; i < num_hidden_layers - 1; ++i)
    {
        matrix_multiply(hiddenLayerOutputs[i], hiddenLayerWeights[i], hiddenLayerInputs[i+1]);
        hiddenLayerInputs[i+1].mapto(activationFunction, hiddenLayerOutputs[i+1]);
    }
    matrix_multiply(hiddenLayerOutputs.back(), outputWeights, outputValues);
    outputValues.print();
}

void Network::back_propagation() {

}

void Network::getLoss() {

}

double relU(double sum) {
    return std::max(0.0, sum);
}
