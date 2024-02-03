#include "NNClassifier.h"

#include "iostream"
#include <cmath>

NNClassifier::NNClassifier(int hidden_layers, int depth, ActivationFunctionPointer activation_function, ActivationFunctionPointer activation_function_derivative)
        :num_hidden_layers{hidden_layers},
         depth{depth},
         activationFunction {activation_function},
         activationFunctionDerivative{activation_function_derivative}
{
}

void NNClassifier::init(std::vector<double> &input, std::vector<double> &output) {
    //input/output
    inputValues = Matrix(input.size(), 1, input);
    outputLayerInputs = Matrix(1, output.size());
    outputLayerOuputs = Matrix(1, output.size());
    targetValues = Matrix(1, output.size(), output);
    outputLayerError = Matrix(output.size(), 1);

    outputLayerDerivatives = Matrix(1,output.size());

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

void NNClassifier::feed_forward() {
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
    matrix_multiply(hiddenLayerOutputs.back(), outputWeights, outputLayerInputs);
    outputLayerInputs.mapto(activationFunction, outputLayerOuputs);
    outputLayerOuputs.print();

    //calculate derivatives
    for (int i = 0; i < num_hidden_layers; ++i) {
        for (int j = 0; j < hiddenLayerDerivatives[i].getsize(); ++j) {
            hiddenLayerDerivatives[i].assign(0,j, activationFunctionDerivative(hiddenLayerOutputs[i].getval(0,j)));
        }
    }
    for (int i = 0; i < targetValues.getsize(); ++i) {
        outputLayerDerivatives.assign(0,i, activationFunctionDerivative(outputLayerOuputs.getval(0,i)));
    }
}

void NNClassifier::back_propagate() {
    //output layer
    outputLayerError = targetValues - outputLayerOuputs;
    outputLayerError.elementwiseMultiply(outputLayerDerivatives);
    hiddenLayerOutputs.back().oneDimentionalTranspose();
    matrix_multiply(hiddenLayerOutputs.back(), outputLayerError, outputWeightPartials);
    hiddenLayerOutputs.back().oneDimentionalTranspose();

    //first hidden layer (at the back)
    outputWeightPartials.transpose();
    matrix_multiply(outputLayerError,outputWeightPartials, hiddenLayerErrors.back());
    outputWeightPartials.transpose();
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

void NNClassifier::updateLoss() {
    double m = 1; //need to set m for when actually training multiple instances
    Matrix temp {targetValues-outputLayerOuputs};
    double sum = 0;
    for (double x: temp.getVector()) {
        sum += 1/m * (x*x);
    }
    currentLoss = (-0.5) * sum;
}

void NNClassifier::updateWeights(double learningRate) {
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

void NNClassifier::checkStatus() {
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

Matrix NNClassifier::getOutput() {
    return outputLayerOuputs;
}

