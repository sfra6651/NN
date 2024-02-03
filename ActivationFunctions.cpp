#include "ActivationFunctions.h"

double relU(double sum) {
    return std::max(0.0, sum);
}

double relUDerivative(double x){
    if (x > 0.0) {return 1.0;}
    return 0.0;
}

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of the sigmoid function
double sigmoidDerivative(double x) {
    double sx = sigmoid(x);
    return sx * (1 - sx);
}