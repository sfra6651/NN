#include <vector>
#include <iostream>
#include "MnistLoader.h"





void MnistLoaderTest(){
    int image = 16;
    int numTrainingImages = 100;
    int numTestImages = 100;
    MnistLoader mnistLoader(numTrainingImages, numTestImages);
    std::cout << "Label: " << mnistLoader.testLabels[image] << "\n";
    for (int i = 0; i < 28; ++i) {
        for(int j = 0; j < 28; ++j) {
            if(mnistLoader.testImages[image][28*i + j] > 0) {
                if(mnistLoader.testImages[image][28*i + j] > 128) {
                    std::cout << "0";
                } else {
                    std::cout << "-";
                }
            } else {
                std::cout << " ";
            }
        }
        std::cout << "\n";
    }

}

void runTests(){
    MnistLoaderTest();
}
