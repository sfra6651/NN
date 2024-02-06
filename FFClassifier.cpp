#include "FFClassifier.h"
#include "Matrix.h"

FFClassifier::FFClassifier(std::vector<int> sizes) {
    if (sizes.size() < 3) {
        throw std::runtime_error("FFClassifier input must be at least length 3");
    }
    layers = sizes.size();
    for (int i = 1; i < sizes.size(); ++i) {
        inputs.push_back(Matrix(sizes[i], 1));
        errors.push_back(Matrix(sizes[i], 1));
        biases.push_back(Matrix(sizes[i], 1, true));
        derivatives.push_back(Matrix(sizes[i], 1));
        outputs.push_back(Matrix(sizes[i], 1));
        weights.push_back(Matrix(sizes[i], sizes[i-1],true));
    }
}

void FFClassifier::train(MnistLoader &loader) {
    int batchSize = 10;
    int batches = loader.trainingImages.size()/batchSize;
    for (int i = 0; i < batches; ++i) {
        std::vector<Matrix> partials;
        for (int j = 0; j < batchSize; ++j) {
            Matrix inputValues(784,1, loader.trainingImages[i*batchSize + j]);
            feedForward(inputValues);
            //backprop(partials);
        }
        update(partials);
    }
}


void FFClassifier::feedForward(Matrix& values) {

    for (int i = 0; i < layers - 1; ++i) {
        if (i==0) {
            matrix_multiply(weights[i], values, inputs[i]);
        } else {
            matrix_multiply(weights[i], outputs[i-1], inputs[i]);
        }
        inputs[i].add(biases[i]);
        inputs[i].mapto(sigmoid, outputs[i]);
        inputs[i].mapto(sigmoidDerivative, derivatives[i]);
    }
}

void FFClassifier::backProp(std::vector<Matrix> &partials) {
    for (int i = layers - 1; i > -1; --i) {
        if (i == layers-1) {
            errors[i] = targets - outputs[i];
            errors[i].elementwiseMultiply(derivatives[i]);
        }
    }
}

void FFClassifier::update(std::vector<Matrix> &partials) {

}






