#ifndef LAYER_H
#define LAYER_H
#include "Matrix.h"
#include <vector>

using Vector = std::vector<double>;

class Layer {
    Vector _activations {};
    Vector _biases {};
    Matrix _weights {};
public:
    Layer(size_t in, size_t out, const Vector &input = {0.0});
    void forward(Vector &input);
    void updateWeight(int row, int col, double gradient);
    Vector getWeight(int neuron) const;

    Vector & getActivations();
    size_t previousLayerSize() const;
    static Vector toSigmoid(const Vector &raw);
    friend Vector operator*(Vector &activations, const Matrix &weights);
    friend Vector operator+(const Vector &product, const Vector &biases);
};

#endif //LAYER_H