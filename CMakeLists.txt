cmake_minimum_required(VERSION 3.27)
project(NN)

set(CMAKE_CXX_STANDARD 17)

add_executable(NN main.cpp
        tests.cpp
        ActivationFunctions.h
        ActivationFunctions.cpp
        MnistLoader.cpp
        MnistLoader.h
)
