#include "../../inc/nn/Layer.h"
#include <cmath>
#include <random>


Layer::Layer(size_t in, size_t out, const Vector &input)
    : _activations(input), _weights(in, out) {
    _biases.assign(out, 0.0);
}
// input = x1000 (1 * w11 + 2 * w21 + 3 * w31) cols = 128, rows 1000
Vector operator*(Vector &activations, const Layer::Matrix &weights) {
    size_t out_size = weights.cols();
    size_t in_size = weights.rows();
    // activations.assign(in_size, 0.0);
    Vector dot_product(out_size);
    for (int i = 0; i < out_size; ++i) {
        Vector column = weights.column(i);
        for (int j = 0; j < in_size; ++j)
            dot_product[i] += activations[j] * column[j]; // 1 x 128
    }
    return dot_product;
}
Vector operator+(const Vector &product, const Vector &biases) {
    const size_t layer_size = product.size();
    Vector pre_activation(layer_size);
    for (int i = 0; i < layer_size; ++i)
        pre_activation[i] = product[i] + biases[i];
    return pre_activation;
}

void Layer::forward(Vector &input) {
    _activations = toSigmoid(input * _weights + _biases);
}

void Layer::updateWeight(int row, int col, double gradient){
    _weights.updateWeight(row, col, gradient);
}

Vector Layer::getWeight(int neuron) const {
  return _weights.row(neuron);
}

Vector & Layer::getActivations() {
    return _activations;
}

size_t Layer::previousLayerSize() const {
    return _weights.rows();
}

Vector Layer::toSigmoid(const Vector &logits) {
    Vector activated;
    activated.reserve(logits.size());
    for (const double node : logits)
        activated.push_back(1 / (1 + exp(-node)));
    return activated;
}

/*  Nested class Matrix  */

Layer::Matrix::Matrix() = default;

Layer::Matrix::Matrix(const size_t in_layer, const size_t out_layer)
    : _row(in_layer), _col(out_layer), _data(_row, Vector(_col)) {
    xavierNormal();
}

size_t Layer::Matrix::rows() const {
    return _row;
}

size_t Layer::Matrix::cols() const {
    return _col;
}
void Layer::Matrix::updateWeight(int row, int col, double gradient) {
    _data[row][col] -= gradient;
}

Vector Layer::Matrix::column(int neuron) const {
    Vector connected_weights(_row);
    for (int weight = 0; weight < _row; ++weight)
        connected_weights[weight] = _data[weight][neuron];
    return connected_weights;
}

Vector Layer::Matrix::row(int neuron) const {
    Vector connected_weights(_col);
    for (int weight = 0; weight < _col; ++weight)
        connected_weights[weight] = _data[neuron][weight];
    return connected_weights;
}

void Layer::Matrix::xavierNormal() {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0, sqrt(2.0 / static_cast<double>(_row + _col))};
    auto random_value = [&d, &gen] { return d(gen); };
    for (int i = 0; i < _row; ++i) {
        for (int j = 0; j < _col; ++j)
            _data[i][j] = random_value();
    }
}


