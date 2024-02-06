#ifndef NN_FFCLASSIFIER_H
#define NN_FFCLASSIFIER_H


#include "Matrix.h"
#include "ActivationFunctions.h"
#include "MnistLoader.h"

class FFClassifier {
public:
    FFClassifier(std::vector<int> sizes);
    void train(MnistLoader& loader);
    void feedForward(Matrix& values);
    void backProp(std::vector<Matrix>& partials);
    void update(std::vector<Matrix>& partials);
    double learningRate = 3.0;
    int layers;

    Matrix targets;
    std::vector<Matrix> inputs;
    std::vector<Matrix> outputs;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
    std::vector<Matrix> errors;
    std::vector<Matrix> derivatives;

};


#endif //NN_FFCLASSIFIER_H
