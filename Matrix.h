#ifndef NN_MATRIX_H
#define NN_MATRIX_H
#include <vector>
#include <random>


class Matrix {
public:
    Matrix();
    Matrix(int x, int y, bool random = false);
    Matrix(int x, int y, const std::vector<double>& invec);
    Matrix(std::vector<double>& left, std::vector<double>& rhs);

    double getval(int x, int y);
    int getrows() const;
    int getcols() const;
    std::vector<double>& getVector();
    int getsize();

    void assign(int row, int col, double val);
    void print();
    using ActivationFunctionPointer = double (*)(double);
    void map(ActivationFunctionPointer func);
    void mapto(ActivationFunctionPointer func, Matrix &input) const;
    void add(Matrix& other);
    void subtract(Matrix& other);
    Matrix operator -(Matrix& other) const;
    bool operator ==(Matrix& other) const;
    void scalarMultiply(double scalar);
    void elementwiseMultiply(Matrix &left, Matrix& right);
    void transpose();
    void oneDimentionalTranspose();
    void zero();

private:
    int size;
    int rows;
    int cols;
    std::vector<double> vec;
};

void matrix_multiply(Matrix &a, Matrix &b, Matrix& output);


#endif //NN_MATRIX_H
