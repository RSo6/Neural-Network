#include "../../inc/nn/Trainer.h"

Trainer::Trainer(double lr) : _learning_rate(lr) {}

void Trainer::backward(Layer &layer,  const Vector &previous, const Vector &expected) {
    const Vector & real = layer.getActivations();
    _deltaSize = real.size();
    _gradientSize = layer.previousLayerSize();
    _delta.assign(_gradientSize, 0.0);
    _post_delta.assign(_deltaSize, 0.0);
    Vector weights(_deltaSize);
    for (int i = 0; i < _deltaSize; ++i) {
        _post_delta[i] = 2 * (expected[i] - real[i]) * real[i] * (1 - real[i]);
        weights = layer.getWeight(i);
        for (int j = 0; j < _gradientSize; ++j)
            _delta[j] += _post_delta[i] * weights[j];
    }
    for (int j = 0; j < _gradientSize; ++j)
        _delta[j] *= previous[j] * (1 - previous[j]);
}

void Trainer::gradientDescent(Layer &layer, const Vector &previous) const {
    for (int i = 0; i < _deltaSize; ++i) {
        for (int j = 0; j < _gradientSize; ++j)
            layer.updateWeight(i, j, _learning_rate*(_post_delta[i] * previous[j]));
    }
}




