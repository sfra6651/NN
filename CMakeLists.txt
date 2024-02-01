cmake_minimum_required(VERSION 3.27)
project(NN)

set(CMAKE_CXX_STANDARD 17)

add_executable(NN main.cpp
        Network.cpp
        Network.h
        Neuron.cpp
        Neuron.h
        Matrix.cpp
        Matrix.h
        tests.cpp)
