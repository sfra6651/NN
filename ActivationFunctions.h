#ifndef NN_ACTIVATIONFUNCTIONS_H
#define NN_ACTIVATIONFUNCTIONS_H

#include <algorithm>

double relU(double sum);
double relUDerivative(double x);

// Sigmoid activation function
double sigmoid(double x);

// Derivative of the sigmoid function
double sigmoidDerivative(double x);
#endif //NN_ACTIVATIONFUNCTIONS_H
