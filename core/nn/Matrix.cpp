#include "../../inc/nn/Matrix.h"
#include <cmath>
#include <random>

Matrix::Matrix() = default;

Matrix::Matrix(const size_t in_layer, const size_t out_layer)
    : _row(in_layer), _col(out_layer), _data(_row, vector<double>(_col)) {
    xavierNormal();
}

size_t Matrix::rows() const {
    return _row;
}

size_t Matrix::cols() const {
    return _col;
}
void Matrix::updateWeight(int row, int col, double gradient) {
    _data[row][col] -= gradient;
}

vector<double> Matrix::column(int neuron) const {
    vector<double> connected_weights(_row);
        for (int weight = 0; weight < _row; ++weight)
            connected_weights[weight] = _data[weight][neuron];
    return connected_weights;
}

vector<double> Matrix::row(int neuron) const {
    vector<double> connected_weights(_col);
        for (int weight = 0; weight < _col; ++weight)
            connected_weights[weight] = _data[neuron][weight];
    return connected_weights;
}

void Matrix::xavierNormal() {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, sqrt(2.0 / static_cast<double>(_row + _col))};
    auto random_value = [&d, &gen] { return d(gen); };
    for (int i = 0; i < _row; ++i) {
        for (int j = 0; j < _col; ++j)
            _data[i][j] = random_value();
    }
}

