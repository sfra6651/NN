#ifndef NN_MNISTLOADER_H
#define NN_MNISTLOADER_H


#include <fstream>
#include <iostream>
#include <vector>

struct Image {
    std::vector<double> pixels;
};

class MnistLoader {
public:
    std::vector<Image> trainingImages;
    std::vector<double> trainingLabels;
    std::vector<Image> testImages;
    std::vector<double> testLabels;

    MnistLoader(int numTraining, int numTest);

private:
    const int IMAGE_SIZE = 28 * 28;

    void loadTrainingData(int count);
    void loadTestData(int count);

    std::vector<Image> readImages(const std::string &filename, int count);
    std::vector<double> readLabels(const std::string &filename, int count);
};


#endif //NN_MNISTLOADER_H
