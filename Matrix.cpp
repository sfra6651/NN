#include "Matrix.h"
#include <iostream>
#include <stdexcept>

Matrix::Matrix()
    : rows{0},
    cols{0}
{
}


Matrix::Matrix(int i, int j, bool random, double n)
    : rows{i},
    cols{j},
    size{i*j}
{
    if (random){
        std::default_random_engine engine(42);
        std::normal_distribution<double> distribution(0.0, std::sqrt(1.0/n)); ;
        for(int x = 0; x < i*j; ++x) { vec.push_back(distribution(engine)); }
    }
    else {
        for(int x = 0; x < i*j; ++x) { vec.push_back(0.0); }
    }
}

Matrix::Matrix(int x, int y, double val)
    : rows{x},
      cols{y},
      size{x*y}
{
    for(int i = 0; i < x*y; ++i) { vec.push_back(val); }
}

Matrix::Matrix(std::vector<std::vector<double>>& input) {
    //each inner vector is appended as a row.
    cols = input[0].size();
    rows = input.size();
    size = rows*cols;
    for(int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            vec.push_back(input[i][j]);
        }
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

Matrix Matrix::operator*(Matrix &left) {
    if (cols != left.getrows()){
        throw std::runtime_error("Incompatible matrices");
    }
    Matrix output(cols,left.getrows());

    for (int i = 0; i < rows; ++i) //row in this
    {
        for (int j = 0; j < left.getcols(); ++j)//col in left
        {
            double sum = 0;
            for(int k = 0; k < cols; ++k)// compute dot prod of the 2 rows
            {
                sum += this->getval(i, k) * left.getval(k,j);
            }
            output.assign(i, j, sum);
        }
    }

    return output;
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

void Matrix::columnWiseSubtract(Matrix& left, Matrix& right) {
    //specifically for computing error in output layer. for every col in left subtract right and replace
    // the corresponding col in this with those values
    if (rows != left.rows || rows != right.rows || cols != left.cols) {
        throw std::runtime_error("attempting to do columnWiseSubtract, matrices are of wrong size");
    }
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j) {
            double temp = right.getval(j,0) - left.getval(i,j);
            this->assign(i,j, temp);
        }
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

void Matrix::transposeInplace() {
    // Create a temporary matrix to hold the transposed values
    std::vector<double> tempVec = vec; // Copy the original data
    int oldRows = rows;
    int oldCols = cols;

    // Swap the dimensions for the current matrix
    rows = oldCols;
    cols = oldRows;

    // Use the temporary vector to read original values and transpose them into the current matrix
    for (int i = 0; i < oldRows; ++i) {
        for (int j = 0; j < oldCols; ++j) {
            // Note: 'assign' and 'getval' should correctly handle indexing within the flattened vector
            this->assign(j, i, tempVec[i * oldCols + j]); // Transpose logic
        }
    }
}


Matrix Matrix::getTranspose() {
    Matrix temp(cols,rows);
    for (size_t i = 0; i < cols; ++i) {
        for (size_t j = 0; j < rows; ++j) {
            temp.assign(i, j, this->getval(j, i));
        }
    }

    return temp;
}

int Matrix::getsize() {
    return size;
}

void Matrix::oneDimentionalTranspose() {
    int tempcols = cols;
    cols = rows;
    rows = tempcols;
}

double Matrix::elementSum() {
    double sum = 0;
    for (auto &x: vec) {
        sum += x;
    }
}

void Matrix::softmax(Matrix &input) {
    this->transposeInplace();//don't need to transpose but makes brain easier
    Matrix temp = input.getTranspose();
    for (int i = 0; i < rows; ++i){
        double sum = 0.0;
        for (int k = 0; k < cols; ++k) {
            sum += std::exp(temp.getval(i,k));
        }
        for (int j = 0; j < cols; ++j) {
            vec[i*cols + j] = std::exp(temp.getval(i,j)) / sum;
        }

    }
    this->transposeInplace();
}

// Function to L2 normalize each column
void Matrix::normalizeColumns() {
    for (int j = 0; j < cols; ++j) { // For each column
        double sumSq = 0.0;
        for (int i = 0; i < rows; ++i) { // Compute the L2 norm for the column
            double value = this->getval(i, j);
            sumSq += value * value;
        }
        double l2Norm = std::sqrt(sumSq);

        if (l2Norm != 0) { // Avoid division by zero
            for (int i = 0; i < rows; ++i) { // Normalize each element in the column
                double value = this->getval(i, j);
                this->assign(i, j, value / l2Norm);
            }
        }
    }
}

int Matrix::count() {
    int c = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (vec[i*cols + j] > 0.0) {
                ++c;
            }
        }
    }
    return c;
}







