#include "NNRegressor.h"
#include "iostream"
#include <cmath>

NNRegressor::NNRegressor(int hidden_layers, int depth, ActivationFunctionPointer activation_function, ActivationFunctionPointer activation_function_derivative)
    :num_hidden_layers{hidden_layers},
    depth{depth},
    activationFunction {activation_function},
    activationFunctionDerivative{activation_function_derivative}
{
}

void NNRegressor::init(std::vector<double> &input, std::vector<double> &output) {
    //input/output
    inputValues = Matrix(input.size(), 1, input);
    outputValues = Matrix(1, output.size(), output);
    targetValues = Matrix(1, output.size(), output);
    outputError = Matrix(output.size(), 1);

    //weights and partials
    inputWeights = Matrix(input.size(), depth, true);
    inputWeightPartials = Matrix(input.size(), depth);
    outputWeights = Matrix(depth,output.size(), true);
    outputWeightPartials = Matrix(depth, output.size());
    for (int i = 0; i < num_hidden_layers - 1;  ++i) {
        hiddenLayerWeights.push_back(Matrix(depth, depth, true));
        hiddenLayerPartials.push_back(Matrix(depth,depth));
    }

    //hidden layer nodes
    for (int i = 0; i < num_hidden_layers;  ++i) {
        hiddenLayerOutputs.push_back(Matrix(1,depth));
        hiddenLayerInputs.push_back(Matrix(1,depth));
        hiddenLayerDerivatives.push_back(Matrix(1,depth));
        hiddenLayerErrors.push_back(Matrix(1,depth));
    }
}

void NNRegressor::feed_forward() {
    //input layer
    inputValues.oneDimentionalTranspose();
    matrix_multiply(inputValues, inputWeights, hiddenLayerInputs[0]);
    inputValues.oneDimentionalTranspose();
    hiddenLayerInputs[0].mapto(activationFunction, hiddenLayerOutputs[0]);

    //hidden layers
    for(int i = 0; i < num_hidden_layers - 1; ++i)
    {
        matrix_multiply(hiddenLayerOutputs[i], hiddenLayerWeights[i], hiddenLayerInputs[i+1]);
        hiddenLayerInputs[i+1].mapto(activationFunction, hiddenLayerOutputs[i+1]);
    }

    //output layer
    matrix_multiply(hiddenLayerOutputs.back(), outputWeights, outputValues);
    outputValues.print();

    //calculare hidden layer derivatives
    for (int i = 0; i < num_hidden_layers; ++i) {
        for (int j = 0; j < hiddenLayerDerivatives[i].getsize(); ++j) {
            hiddenLayerDerivatives[i].assign(0,j, activationFunctionDerivative(hiddenLayerOutputs[i].getval(0,j)));
        }
    }
}

void NNRegressor::back_propagate() {
    //output layer
    outputError = targetValues - outputValues;
    hiddenLayerOutputs.back().oneDimentionalTranspose();
    matrix_multiply(hiddenLayerOutputs.back(), outputError, outputWeightPartials);
    hiddenLayerOutputs.back().oneDimentionalTranspose();

    //first hidden layer (at the back)
    outputWeightPartials.transposeInplace();
    matrix_multiply(outputError,outputWeightPartials, hiddenLayerErrors.back());
    outputWeightPartials.transposeInplace();
    hiddenLayerErrors.back().elementwiseMultiply(hiddenLayerDerivatives.back());


    //middle layers
    for (int i = num_hidden_layers - 2; i > -1; --i) {
        hiddenLayerErrors[i+1].oneDimentionalTranspose();
        matrix_multiply(hiddenLayerErrors[i+1], hiddenLayerOutputs[i+1], hiddenLayerPartials[i]);
        hiddenLayerErrors[i+1].oneDimentionalTranspose();
        matrix_multiply(hiddenLayerOutputs[i], hiddenLayerPartials[i], hiddenLayerErrors[i]);
        hiddenLayerErrors[i].elementwiseMultiply(hiddenLayerDerivatives[i]);
    }

    //final hidden layer (at the front)

    //input layer
    matrix_multiply(inputValues, hiddenLayerErrors[0], inputWeightPartials);
}

void NNRegressor::updateLoss() {
    double m = 1; //need to set m for when actually training multiple instances
    Matrix temp {targetValues-outputValues};
    double sum = 0;
    for (double x: temp.getVector()) {
        sum += 1/m * (x*x);
    }
    currentLoss = (-0.5) * sum;
}

void NNRegressor::updateWeights(double learningRate) {
    outputWeightPartials.scalarMultiply(learningRate);
    inputWeightPartials.scalarMultiply(learningRate);

    outputWeights.add(outputWeightPartials);
    inputWeights.add(inputWeightPartials);

    for (auto& x: hiddenLayerPartials) {
        x.scalarMultiply(learningRate);
    }
    int index = 0;
    for (auto& x: hiddenLayerWeights) {
        x.add(hiddenLayerPartials[index]);
        ++index;
    }

}

void NNRegressor::checkStatus() {
//    std::cout << "Output layer \n";
//    std::cout << "Weights: " << outputWeights.getrows() << " x " << outputWeights.getcols() <<":\n";
//    std::cout << "partial derivatives:" << outputWeightPartials.getrows() << " x " << outputWeightPartials.getcols() <<":\n";
//
//    for (int i = 0; i < num_hidden_layers -1; ++i) {
//        std::cout << "\n layer " << i << ":\n";
//        std::cout << "Weights: " << hiddenLayerWeights[i].getrows() << " x " << hiddenLayerWeights[i].getcols() <<":\n";
//        std::cout << "partial derivatives:" << hiddenLayerPartials[i].getrows() << " x " << hiddenLayerPartials[i].getcols() <<":\n";
//
//    }
//
//    std::cout << "\nInput layer \n";
//    std::cout << "Weights: " << inputWeights.getrows() << " x " << inputWeights.getcols() <<":\n";
//    std::cout << "partial derivatives:" << inputWeightPartials.getrows() << " x " << inputWeightPartials.getcols() <<":\n";

    std::cout << "input weights\n";
    inputWeights.print();
    std::cout << "output weights\n";
    outputWeights.print();

}

Matrix NNRegressor::getOutput() {
    return outputValues;
}

