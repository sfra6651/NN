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
    inputValues = Matrix(input.size(), 1, input);
    outputValues = Matrix(1, output.size(), output);
    targetValues = Matrix(1, output.size(), output);
    outputError = Matrix(output.size(), 1);

    inputWeights = Matrix(input.size(), depth, true);
    inputWeightDeltas = Matrix(input.size(), depth);
    outputWeights = Matrix(depth,output.size(), true);
    outputWeightDeltas = Matrix(depth, output.size());

//    std::cout << "input/out weights assinged\n";

    for (int i = 0; i < num_hidden_layers - 1;  ++i) {
        hiddenLayerWeights.push_back(Matrix(depth, depth, true));
    }
    for (int i = 0; i < num_hidden_layers;  ++i) {
        hiddenLayerOutputs.push_back(Matrix(1,depth));
        hiddenLayerInputs.push_back(Matrix(1,depth));
        hiddenLayerErrors.push_back(Matrix(1,depth));
        hiddenLayerDerivatives.push_back(Matrix(1,depth));
    }
}

void Network::feed_forward() {
    inputValues.oneDimentionalTranspose();

    matrix_multiply(inputValues, inputWeights, hiddenLayerInputs[0]);
    hiddenLayerInputs[0].mapto(activationFunction, hiddenLayerOutputs[0]);

    for(int i = 0; i < num_hidden_layers - 1; ++i)
    {
        matrix_multiply(hiddenLayerOutputs[i], hiddenLayerWeights[i], hiddenLayerInputs[i+1]);
        hiddenLayerInputs[i+1].mapto(activationFunction, hiddenLayerOutputs[i+1]);
    }
    matrix_multiply(hiddenLayerOutputs.back(), outputWeights, outputValues);
    outputValues.print();
    //calculare hidden layer derivatives
    for (int i = 0; i < num_hidden_layers; ++i) {
        for (int j = 0; j < hiddenLayerDerivatives[i].getsize(); ++j) {
            hiddenLayerDerivatives[i].assign(0,j, relUDerivative(hiddenLayerOutputs[i].getval(0,j)));
        }
    }
    inputValues.oneDimentionalTranspose();
}

void Network::back_propagate() {
    hiddenLayerOutputs.back().oneDimentionalTranspose();
    outputError.oneDimentionalTranspose();

    outputError = targetValues - outputValues;
    matrix_multiply(hiddenLayerOutputs.back(), outputError, outputWeightDeltas);

    hiddenLayerOutputs.back().oneDimentionalTranspose();

    outputWeights.transpose();

    matrix_multiply(outputError, outputWeights, hiddenLayerErrors[num_hidden_layers-1]);

    for (int i = num_hidden_layers - 2; i > -1; --i) {
        hiddenLayerWeights[i].transpose();
        matrix_multiply(hiddenLayerErrors[i+1], hiddenLayerWeights[i], hiddenLayerErrors[i]);
    }
    matrix_multiply(inputValues, hiddenLayerErrors[0], inputWeightDeltas);
}

void Network::updateLoss() {
    double m = 1; //need to set m for when actually training multiple instances
    Matrix temp {targetValues-outputValues};
    double sum = 0;
    for (double x: temp.getVector()) {
        sum += 1/m * (x*x);
    }
    currentLoss = (-0.5) * sum;
}

double relU(double sum) {
    return std::max(0.0, sum);
}

double relUDerivative(double x){
    if (x > 0.0) {return x;}
    return 0.0;
}
