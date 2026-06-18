#ifndef LAYER_H
#define LAYER_H
#include "Matrix.h"
#include <vector>

using Vector = std::vector<double>;

class Layer {
    size_t out_layer_size {};
    Vector _activations {};
    Vector _biases {};
    Matrix _weights;
public:
    Layer();
    Vector forward(const Vector & input);
    Vector backward();
    double activation(double neuron);
    Vector operator*(const Matrix & weights);
    friend Vector operator*(const Vector & activations, const Matrix & weights);
};

#endif //LAYER_H