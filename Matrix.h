#ifndef NN_MATRIX_H
#define NN_MATRIX_H
#include <vector>
#include <random>


class Matrix {
public:
    Matrix();
    Matrix(int x, int y, double val);
    Matrix(int x, int y, bool random = false, double n = 1.0);
    Matrix(int x, int y, const std::vector<double>& invec);
    Matrix(std::vector<double>& left, std::vector<double>& rhs);
    explicit Matrix(std::vector<std::vector<double>>& input);

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
    void columnWiseSubtract(Matrix& left, Matrix& right);
    bool operator ==(Matrix& other) const;
    void scalarMultiply(double scalar);
    void elementwiseMultiply(Matrix &left, Matrix& right);
    void elementwiseMultiply(Matrix &right);
    void transposeInplace();
    Matrix getTranspose();
    void oneDimentionalTranspose();
    void zero();
    void softmax(Matrix& input);
    Matrix operator*(Matrix& left);
    double elementSum();
    int count();
    void normalizeColumns();

private:
    int size{};
    int rows{};
    int cols{};
    std::vector<double> vec;
};

void matrix_multiply(Matrix &a, Matrix &b, Matrix& output);


#endif //NN_MATRIX_H
