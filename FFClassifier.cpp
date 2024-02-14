#include "FFClassifier.h"
#include "Matrix.h"

std::vector<double> setTargetVector(double target) {
    std::vector<double> vector(10);
    for (auto &x: vector) {
        x = 0;
    }
    vector[static_cast<int>(target)] = 1.0;
    return  vector;
}

FFClassifier::FFClassifier(std::vector<int> sizes, int batchSize) {
    if (sizes.size() < 3) {
        throw std::runtime_error("FFClassifier input must be at least length 3");
    }
    layers = sizes.size();
    m = batchSize;
    for (int i = 1; i < sizes.size(); ++i) {
        inputs.push_back(Matrix(sizes[i], m));
        errors.push_back(Matrix(sizes[i], m));
        biases.push_back(Matrix(sizes[i], m, true));
        biasPartials.push_back(Matrix(sizes[i], m));
        derivatives.push_back(Matrix(sizes[i], m));
        outputs.push_back(Matrix(sizes[i], m));
        weights.push_back(Matrix(sizes[i], sizes[i-1], true, sizes[i-1]));
        weightPartials.push_back(Matrix(sizes[i], sizes[i-1]));
    }
}

void FFClassifier::train(MnistLoader &loader) {
    int batches = loader.trainingImages.size()/m;
//    std::cout << batches << "\n";

    for (int  i = 0; i < batches - 1; ++i) {
//        std::cout << "starting batch: " << i << "\n";
        auto start = loader.trainingImages.begin() + (i*m);
        auto end = loader.trainingImages.begin() + (i*m + m);
        std::vector<std::vector<double>> slice(start, end);
//        for (auto& x: slice) {
//            normalizeVector(x);
//        }
        Matrix values(slice);
        values.scalarMultiply(1.0/255.0);
        values.transposeInplace();//in constructor each image is assigned to a row, transposeInplace to have each image as a col instead.
        feedForward(values);

        //seting target matrix
        auto targetStart = loader.trainingLabels.begin() + (i*m);
        auto targetEnd = loader.trainingLabels.begin() + (i*m + m);
        std::vector<double> t(targetStart, targetEnd);
        std::vector<std::vector<double>> tg;
        for (auto &x: t) {
            tg.push_back(setTargetVector(x));
        }
        Matrix targets(tg);
        targets.transposeInplace();
        backProp(values, targets);

//        for (auto &x: t) {
//            std::cout << x << " ";
//        }
//        std::cout << "\n";
    }
//    outputs[1].print();
}


void FFClassifier::feedForward(Matrix& values) {
    for (int i = 0; i < layers - 1; ++i) {
        if (i==0) {
            matrix_multiply( weights[i], values, inputs[i]);
        } else {
            matrix_multiply( weights[i], outputs[i-1], inputs[i]);
        }
        inputs[i].add(biases[i]);
//        inputs[i].normalizeColumns();
        inputs[i].mapto(sigmoid, outputs[i]);
        inputs[i].mapto(sigmoidDerivative, derivatives[i]);
    }
//    outputs.back().print();
}

void FFClassifier::backProp(Matrix& values, Matrix& targets) {
    for (int i = layers - 2; i > -1; --i) {
        //check layer for error calculation. last(output) layer is different from hidden layers
        if (i == layers-2) {
            errors[i] = outputs[i] - targets;
//            errors[i].elementwiseMultiply(derivatives[i]);
        } else {
            Matrix WTranspose = weights[i+1].getTranspose();
            matrix_multiply(WTranspose, errors[i+1], errors[i]);
            errors[i].elementwiseMultiply(derivatives[i]);
        }

        // check layer for weightPartials calculation. first layer uses input instead of output from previous layer
        if (i > 0) {
            Matrix ATranspose = outputs[i-1].getTranspose();
            matrix_multiply(errors[i], ATranspose, weightPartials[i]);
            weightPartials[i].scalarMultiply(1.0/static_cast<double>(m));
            double db = errors[i].elementSum() / static_cast<double>(m);
            biasPartials[i] = Matrix(biases[i].getrows(), biases[i].getcols(), db);
        } else {
            Matrix VTranspose = values.getTranspose();
            matrix_multiply(errors[i], VTranspose, weightPartials[i]);
            weightPartials[i].scalarMultiply(1.0/static_cast<double>(m));
            double db = errors[i].elementSum() / static_cast<double>(m);
            biasPartials[i] = Matrix(biases[i].getrows(), biases[i].getcols(), db);
        }
    }
    //update the network
    for (int i = 0; i < layers - 2; ++i) {
        weightPartials[i].scalarMultiply(learningRate);
        biasPartials[i].scalarMultiply(learningRate);
        weights[i].subtract(weightPartials[i]);
        biases[i].subtract(biasPartials[i]);
    }
}

void FFClassifier::update(std::vector<Matrix> &partials) {



}

int getCorrect(Matrix&m , std::vector<double> t){
//    Matrix targets = Matrix(1, t.size(), t);
    int count = 0;
    for (int i = 0; i < m.getrows(); ++i) {
        double max = 0;
        int maxPos = 0;
        for (int j = 0;j < m.getcols(); ++j) {
            if (m.getval(i, j) > max) {
                max = m.getval(i, j);
                maxPos = j;
            }
        }
        if (maxPos == static_cast<int>(t[i])) {
            ++count;
        }
    }
    return count;
}

void FFClassifier::test(MnistLoader &loader) {
    int tests = loader.testImages.size() / m;
    int correct = 0;
    for (int i = 0; i < tests -1; ++i) {
        //setting input
        auto start = loader.trainingImages.begin() + (i*m);
        auto end = loader.trainingImages.begin() + (i*m + m);
        std::vector<std::vector<double>> slice(start, end);
        Matrix values(slice);
        values.scalarMultiply(1.0/255.0);
        values.transposeInplace();

        //seting target matrix
        auto targetStart = loader.trainingLabels.begin() + (i*m);
        auto targetEnd = loader.trainingLabels.begin() + (i*m + m);
        std::vector<double> t(targetStart, targetEnd);
        std::vector<std::vector<double>> tg;
        for (auto &x: t) {
            tg.push_back(setTargetVector(x));
        }
        Matrix targets(tg);

        feedForward(values);
        Matrix prediction = outputs.back().getTranspose();
        correct += getCorrect(prediction, t);
//        std::cout << "Targets: ";
//        for (auto x: t) {
//            std::cout << x << " ";
//        }
//        std::cout <<"\n";
//        targets.print();
//        std::cout << "Predictions: " << "\n";
//        prediction.print();


    }
    double accuracy = static_cast<double>(correct)/ static_cast<double> (loader.trainingImages.size());
    std::cout <<"Accuracy: " << accuracy << "\n";
//    std::cout << correct << " out of " << loader.trainingImages.size() << "\n";
}

void normalizeVector(std::vector<double>& v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / v.size() - mean * mean);

    std::transform(v.begin(), v.end(), v.begin(), [mean, std_dev](double& d) {
        return (d - mean) / std_dev;
    });
}






