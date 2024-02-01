#include "Matrix.h"
#include <iostream>
#include <stdexcept>

Matrix::Matrix()
    : rows{0},
    cols{0}
{
}


Matrix::Matrix(int i, int j, bool random)
    : rows{i},
    cols{j}
{
    if (random){
        std::default_random_engine engine(42);
        std::uniform_real_distribution<double> distribution(0, 0.1);
        for(int x = 0; x < i*j; ++x) { vec.push_back(distribution(engine)); }
        std::cout << "created random(0-1) matrix of size: " << rows << " x " << cols << std::endl;
    }
    else {
//        vec.reserve(i*j);
        for(int x = 0; x < i*j; ++x) { vec.push_back(0.0); }
        std::cout << "created 0's matrix of size: " << rows << " x " << cols << std::endl;
    }
}

Matrix::Matrix(int x, int y, const std::vector<double>& invec)
    : rows{x},
    cols{y}
{
    if (x*y == invec.size()){
        vec.assign(invec.begin(),invec.end());
    } else{throw std::runtime_error("trying to construct a matrix from vector with invalid size arguments");}
//    for (double i : invec) {
//        vec.push_back(i);
//    }
}


double Matrix::getval(int row, int col) {
    if (row > rows || col > cols) {throw std::runtime_error("Matrix access out of bounds");}
    return vec[row*cols + col];
}

void Matrix::assign(int row, int col, double val) {
    if (row > rows || col > cols) {throw std::runtime_error("Matrix access out of bounds");}
    vec[row *cols + col] = val;
}

//leave it up to callers to make sure matrices are compatible, but still catch errors at runtime
void matrix_multiply(Matrix& a, Matrix& b, Matrix& output){
    if (a.getcols() != b.getrows() || (output.getrows() != a.getrows() && output.getcols() != b.getcols())){
        throw std::runtime_error("Incompatible matrices");
    }

    for (int i = 0; i < a.getrows(); ++i) //row in a
    {
        for (int j = 0; j < b.getcols(); ++j)//col in b
        {
            double sum = 0;
            for(int k = 0; k < a.getcols(); ++k)// compute dot prod of the 2 rows
            {
                sum += a.getval(i, k) * b.getval(k,j);
            }
            output.assign(i, j, sum);
        }
    }
}

void Matrix::print() {
    for (int i = 0; i < rows; ++i)
    {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < cols; ++j)
        {
            std::cout << vec[j + i * cols] << " ";
        }
        std::cout << "\n";
    }
}

void Matrix::map(ActivationFunctionPointer func) {
    for(double &x: this->vec) {
        x = func(x);
    }
}

void Matrix::mapto(Matrix::ActivationFunctionPointer func, Matrix& input) const{
    int index = 0;
    for(double x: vec) {
        input.vec[index] = func(x);
        ++index;
    }
}


int Matrix::getrows() const {
    return rows;
}

int Matrix::getcols() const {
    return cols;
}

void Matrix::add(Matrix& other) {
    if(cols != other.cols || rows != other.rows) {
        throw std::runtime_error("attempting to add matrices of different sizes");
    }
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            vec[i*rows + cols] += other.vec[i*rows + cols];
        }
    }
}


