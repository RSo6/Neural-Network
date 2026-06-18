#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

class Matrix {
    size_t _row, _col;
    std::vector<double> data {};
    void xavierInitialization();

public:
    Matrix(): _row(0), _col(0) {}
    Matrix(const size_t in_layer, const size_t out_layer): _row(in_layer), _col(out_layer) {}
    Matrix & operator=(const Matrix & weights);
};
#endif //MATRIX_H
