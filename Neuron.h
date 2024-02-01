#ifndef NN_NEURON_H
#define NN_NEURON_H
#include <algorithm>

enum ActivationFunction {
    RELU
};


struct Neuron {
public:
    Neuron(ActivationFunction function);
    int activationFunction;
    bool active;
    double bias;
    double output;
    double input;

    void activation_f();
};

#endif //NN_NEURON_H
