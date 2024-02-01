#ifndef NN_MATRIX_H
#define NN_MATRIX_H
#include <vector>
#include <random>


class Matrix {
public:
    Matrix();
    Matrix(int x, int y, bool random = false);
    Matrix(int x, int y, const std::vector<double>& invec);

    double getval(int x, int y);
    int getrows() const;
    int getcols() const;

    void assign(int row, int col, double val);
    void print();
    using ActivationFunctionPointer = double (*)(double);
    void map(ActivationFunctionPointer func);
    void mapto(ActivationFunctionPointer func, Matrix &input) const;
    void add(Matrix& other);

private:
    int rows;
    int cols;
    std::vector<double> vec;
};

void matrix_multiply(Matrix& a, Matrix& b, Matrix& output);


#endif //NN_MATRIX_H
