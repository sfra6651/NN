cmake_minimum_required(VERSION 3.27)
project(NN)

set(CMAKE_CXX_STANDARD 17)

add_executable(NN main.cpp
        NNRegressor.cpp
        NNRegressor.h
        Matrix.cpp
        Matrix.h
        tests.cpp
        ActivationFunctions.h
        ActivationFunctions.cpp
        NNClassifier.cpp
        NNClassifier.h
        MnistLoader.cpp
        MnistLoader.h
)
