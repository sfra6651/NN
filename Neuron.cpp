#include "Neuron.h"

//bool activation_f(double sum) {
//    return static_cast<bool>(std::max(0.0, sum));
//}

Neuron::Neuron(ActivationFunction function)
    :activationFunction{function},
    bias{0.01}
    {

}
