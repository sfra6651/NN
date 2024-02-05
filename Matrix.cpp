#include "Matrix.h"
#include <iostream>
#include <stdexcept>

Matrix::Matrix()
    : rows{0},
    cols{0}
{
}


Matrix::Matrix(int i, int j, bool random, double stdev)
    : rows{i},
    cols{j},
    size{i*j}
{
    if (random){
        std::default_random_engine engine(42);
        std::uniform_real_distribution<double> distribution(-stdev, stdev);
        for(int x = 0; x < i*j; ++x) { vec.push_back(distribution(engine)); }
    }
    else {
        for(int x = 0; x < i*j; ++x) { vec.push_back(0.0); }
    }
}

Matrix::Matrix(int x, int y, const std::vector<double>& invec)
    : rows{x},
    cols{y},
    size{x*y}
{
    if (x*y == invec.size()){
        vec.assign(invec.begin(),invec.end());
    } else{throw std::runtime_error("trying to construct a matrix from vector with invalid size arguments");}
}

Matrix::Matrix(std::vector<double> &left, std::vector<double> &rhs)
{
    if(left.size() != rhs.size()){
        throw std::runtime_error("Vectors need to be same size to multiple and form a matrix");
    }
    rows = left.size();
    cols = rhs.size();
    size = rows*cols;
    for(int i = 0; i < left.size(); ++i){
        for(int j = 0; j < rhs.size(); ++j) {
            vec.push_back(left[i] * rhs[j]);
        }
    }
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

    int count = 0;

    for(auto &x: vec){
        x += other.vec[count];
        ++count;
    }
}

//only intended for testing
std::vector<double>& Matrix::getVector() {
    return vec;
}

void Matrix::zero() {
    for(auto &x: vec){
        x = 0;
    }
}

Matrix Matrix::operator-(Matrix &other) const {
    Matrix temp(other.getrows(), other.getcols(), vec);
    temp.subtract(other);
    return temp;
}

void Matrix::subtract(Matrix &other) {
    if(cols != other.cols || rows != other.rows) {
        throw std::runtime_error("attempting to subtract matrices of different sizes");
    }

    int count = 0;

    for(auto &x: vec){
        x -= other.vec[count];
        ++count;
    }
}

bool Matrix::operator==(Matrix &other) const {
    if(cols != other.cols || rows != other.rows || vec.size() != other.vec.size()) {
        return false;
    }
    int count = 0;
    for(auto &x: vec) {
        if (x != other.vec[count]) {
            return false;
        }
        ++count;
    }
    return true;
}

void Matrix::scalarMultiply(double scalar) {
    for(auto &x: vec) {
        x *= scalar;
    }
}

void Matrix::elementwiseMultiply(Matrix &left, Matrix& right) {
    if(cols != left.cols || rows != left.rows || cols != right.cols || rows != right.rows) {
        throw std::runtime_error("attempting to elementwise Multiply matrices of different sizes");
    }
    int count = 0;
    for(auto  &x: vec){
        x = left.getVector()[count] * right.getVector()[count];
    }

}

void Matrix::elementwiseMultiply(Matrix& right) {
    if(cols != right.cols || rows != right.rows) {
        throw std::runtime_error("attempting to elementwise Multiply matrices of different sizes");
    }
    int count = 0;
    for(auto  &x: vec){
        x = x * right.getVector()[count];
        ++count;
    }

}

void Matrix::transpose() {
    int oldCols {cols};
    int oldRows {rows};
    cols = rows;
    rows = oldCols;

    Matrix temp(oldRows, oldCols, vec);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            this->assign(i,j,temp.getval(j, i));
        }
    }
}

int Matrix::getsize() {
    return size;
}

void Matrix::oneDimentionalTranspose() {
    int tempcols = cols;
    cols = rows;
    rows = tempcols;
}



