#ifndef NN_FFCLASSIFIER_H
#define NN_FFCLASSIFIER_H


#include "Matrix.h"
#include "ActivationFunctions.h"
#include "MnistLoader.h"

void normalizeVector(std::vector<double>& v);

class FFClassifier {
public:
    FFClassifier(std::vector<int> sizes, int batchSize);
    void train(MnistLoader& loader);
    void feedForward(Matrix& values);
    void backProp(Matrix& values, Matrix& targets);
    void test(MnistLoader& loader);
    void update(std::vector<Matrix>& partials);
    double learningRate = 3;
    int layers;
    int m;

    std::vector<Matrix> inputs;
    std::vector<Matrix> outputs;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
    std::vector<Matrix> errors;
    std::vector<Matrix> derivatives;
    std::vector<Matrix> weightPartials;
    std::vector<Matrix> biasPartials;

};


#endif //NN_FFCLASSIFIER_H
