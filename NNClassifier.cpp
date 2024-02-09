#include "NNClassifier.h"
#include "ActivationFunctions.h"
#include "iostream"
#include <cmath>

NNClassifier::NNClassifier(int hidden_layers, int depth, ActivationFunctionPointer activation_function, ActivationFunctionPointer activation_function_derivative)
        :num_hidden_layers{hidden_layers},
         depth{depth},
         activationFunction {activation_function},
         activationFunctionDerivative{activation_function_derivative},
         outputLayerActivationF{sigmoid},
         outputLayerActivationFDerivative{sigmoidDerivative}
{
}

void NNClassifier::setOuptupLayerFunction(NNClassifier::ActivationFunctionPointer f,NNClassifier::ActivationFunctionPointer fd)
{
    outputLayerActivationF = f;
    outputLayerActivationFDerivative = fd;

}

void NNClassifier::init(std::vector<double> &input, std::vector<double> &output) {
    //input/output
    inputValues = Matrix(input.size(), 1, input);
    outputLayerInputs = Matrix(1, output.size());
    outputLayerOuputs = Matrix(1, output.size());
    targetValues = Matrix(1, output.size(), output);
    outputLayerError = Matrix(output.size(), 1);
    outputLayerDerivatives = Matrix(1,output.size());
    outputLayerBias = Matrix(Matrix(1,output.size(), 0.01));

    //weights and partials
    inputWeights = Matrix(input.size(), depth, true, input.size());
    inputWeightPartials = Matrix(input.size(), depth);
    outputWeights = Matrix(depth,output.size(), true, depth);
    outputWeightPartials = Matrix(depth, output.size());

    inputBatchWeights = Matrix(input.size(), depth);
    outputBatchWeighs = Matrix(depth,output.size());
    for (int i = 0; i < num_hidden_layers - 1;  ++i) {
        hiddenLayerWeights.push_back(Matrix(depth, depth, true, input.size()));
        hiddenLayerPartials.push_back(Matrix(depth,depth));
        hiddenLayerBatchWeights.push_back(Matrix(depth,depth));
    }

    //hidden layer nodes
    for (int i = 0; i < num_hidden_layers;  ++i) {
        hiddenLayerOutputs.push_back(Matrix(1,depth));
        hiddenLayerInputs.push_back(Matrix(1,depth));
        hiddenLayerDerivatives.push_back(Matrix(1,depth));
        hiddenLayerErrors.push_back(Matrix(1,depth));
        hiddenLayerBiases.push_back(Matrix(1,depth,0.01));
    }
}

void NNClassifier::feed_forward() {
    //input layer
    inputValues.oneDimentionalTranspose();
    matrix_multiply(inputValues, inputWeights, hiddenLayerInputs[0]);
    inputValues.oneDimentionalTranspose();
    hiddenLayerInputs[0].add(hiddenLayerBiases[0]);
    hiddenLayerInputs[0].mapto(activationFunction, hiddenLayerOutputs[0]);

    //hidden layers
    for(int i = 0; i < num_hidden_layers - 1; ++i)
    {
        matrix_multiply(hiddenLayerOutputs[i], hiddenLayerWeights[i], hiddenLayerInputs[i+1]);
        hiddenLayerInputs[i+1].add(hiddenLayerInputs[i+1]);
        hiddenLayerInputs[i+1].mapto(activationFunction, hiddenLayerOutputs[i+1]);
    }

    //output layer
    matrix_multiply(hiddenLayerOutputs.back(), outputWeights, outputLayerInputs);
    outputLayerInputs.add(outputLayerBias);
    outputLayerInputs.mapto(outputLayerActivationF, outputLayerOuputs);
//    outputLayerOuputs.print();

    //calculate derivatives
    for (int i = 0; i < num_hidden_layers; ++i) {
        for (int j = 0; j < hiddenLayerDerivatives[i].getsize(); ++j) {
            hiddenLayerDerivatives[i].assign(0,j, activationFunctionDerivative(hiddenLayerOutputs[i].getval(0,j)));
        }
    }
    for (int i = 0; i < targetValues.getsize(); ++i) {
        outputLayerDerivatives.assign(0,i, outputLayerActivationFDerivative(outputLayerOuputs.getval(0,i)));
    }
}

void NNClassifier::back_propagate(double numSamples) {
    //output layer
    double learningRate = 0.05;
    outputLayerError = targetValues - outputLayerOuputs;
    outputLayerError.elementwiseMultiply(outputLayerDerivatives);
    outputLayerError.scalarMultiply(1.0/numSamples); //normalized
    hiddenLayerOutputs.back().oneDimentionalTranspose();
    matrix_multiply(hiddenLayerOutputs.back(), outputLayerError, outputWeightPartials);
    hiddenLayerOutputs.back().oneDimentionalTranspose();
    Matrix bias(outputLayerBias.getrows(), outputLayerBias.getcols());
    bias.add(outputLayerError);
    bias.scalarMultiply(learningRate);
    outputLayerBias.add(bias);

    //first hidden layer (at the back)
    outputWeightPartials.transposeInplace();
    matrix_multiply(outputLayerError,outputWeightPartials, hiddenLayerErrors.back());
    outputWeightPartials.transposeInplace();
    hiddenLayerErrors.back().elementwiseMultiply(hiddenLayerDerivatives.back());
    Matrix Hbias(hiddenLayerBiases[0].getrows(), hiddenLayerBiases[0].getcols());
    Hbias.add(hiddenLayerErrors.back());
    Hbias.scalarMultiply(learningRate);
    hiddenLayerBiases.back().add(Hbias);



    //middle layers
    for (int i = num_hidden_layers - 2; i > -1; --i) {
        hiddenLayerErrors[i+1].oneDimentionalTranspose();
        matrix_multiply(hiddenLayerErrors[i+1], hiddenLayerOutputs[i+1], hiddenLayerPartials[i]);
        hiddenLayerErrors[i+1].oneDimentionalTranspose();
        matrix_multiply(hiddenLayerOutputs[i], hiddenLayerPartials[i], hiddenLayerErrors[i]);
        hiddenLayerErrors[i].elementwiseMultiply(hiddenLayerDerivatives[i]);
        Hbias.zero();
        Hbias.add(hiddenLayerErrors[i]);
        Hbias.scalarMultiply(learningRate);
        hiddenLayerBiases[i].add(Hbias);
    }

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

void NNClassifier::loadDataPoint(std::vector<double>& input, std::vector<double>& target) {
    inputValues = Matrix(input.size(), 1, input);
    targetValues = Matrix(1,target.size(), target);
    inputValues.scalarMultiply(1.0/255.0);
}

void NNClassifier::batchSingleUpdate() {
    inputBatchWeights.add(inputWeightPartials);
    outputBatchWeighs.add(outputWeightPartials);

    int count = 0;
    for (auto &x: hiddenLayerBatchWeights) {
        x.add(hiddenLayerPartials[count]);
        ++count;
    }
}

void NNClassifier::batchFullUpdate(double learnginRate, double batchSize) {
    double scalar = learnginRate/batchSize;

    inputBatchWeights.scalarMultiply(scalar);
    inputWeights.subtract(inputBatchWeights);

    outputBatchWeighs.scalarMultiply(scalar);
    outputWeights.subtract(outputBatchWeighs);

    int count = 0;
    for (auto &x: hiddenLayerWeights) {
        hiddenLayerBatchWeights[count].scalarMultiply(scalar);
        x.subtract(hiddenLayerBatchWeights[count]);
        ++count;
    }
}

void NNClassifier::printErrors() {
    std::cout << "Errors\n";
    for (int i = 0; i < num_hidden_layers; ++i) {
        std::cout << "Hidden layer " << i << " ";
        hiddenLayerErrors[i].print();
    }
    std::cout << "Output layer ";
    outputLayerError.print();
    std::cout << "\n";

}

void NNClassifier::printOuputs() {
    std::cout << "Outputs\n";
    for (int i = 0; i < num_hidden_layers; ++i) {
        std::cout << "Hidden layer " << i << " ";
        hiddenLayerOutputs[i].print();
    }
    std::cout << "Output layer ";
    outputLayerOuputs.print();
//    std::cout << "\n";
}

void NNClassifier::printInputs() {
    std::cout << "Inputs\n";
    for (int i = 0; i < num_hidden_layers; ++i) {
        std::cout << "Hidden layer " << i << " ";
        hiddenLayerInputs[i].print();
    }
    std::cout << "Output layer ";
    outputLayerInputs.print();
//    std::cout << "\n";
}

void NNClassifier::printTargets() {
    std::cout << "Targets\n";
    targetValues.print();
}



