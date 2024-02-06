#include "MnistLoader.h"

MnistLoader::MnistLoader(int numTraining, int numTest) {
    loadTrainingData(numTraining);
    loadTestData(numTest);
}


void MnistLoader::loadTrainingData(int count) {
    trainingImages = readImages("/Users/shaun/Dev/NN/MNIST/train-images.idx3-ubyte", count);
    trainingLabels = readLabels("/Users/shaun/Dev/NN/MNIST/train-labels.idx1-ubyte", count);
}

void MnistLoader::loadTestData(int count) {
    testImages = readImages("/Users/shaun/Dev/NN/MNIST/t10k-images.idx3-ubyte", count);
    testLabels = readLabels("/Users/shaun/Dev/NN/MNIST/t10k-labels.idx1-ubyte", count);
}

std::vector<std::vector<double>> MnistLoader::readImages(const std::string &filename, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    // Skipping the header
    file.seekg(16);

    std::vector<std::vector<double>> images(count);
    for (int i = 0; i < count; ++i) {
        images[i].resize(IMAGE_SIZE);
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            int c;
            c = file.get();
            images[i][j] = static_cast<double>(c);
        }
    }

    return images;
}

std::vector<double> MnistLoader::readLabels(const std::string &filename, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    // Skipping the header
    file.seekg(8);

    std::vector<double> labels(count);
    for (int i = 0; i < count; ++i) {
            int c;
            c = file.get();
            labels[i] = static_cast<double>(c);
    }
    return labels;
}