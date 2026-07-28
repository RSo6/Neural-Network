#include "../../inc/nn/NeuralNetwork.h"
#include "../../inc/nn/Trainer.h"

NeuralNetwork::NeuralNetwork() : NeuralNetwork(
    {1000, 128, 64, 10}, {}, {}) {}

NeuralNetwork::NeuralNetwork(const vector<int> &topology, const Vector &input, const Vector &expected)
        : _topology(topology), _expected(expected) {
    _layers.emplace_back(_topology[0], _topology[1], input);
    for (int i = 1; i < _topology.size() - 1; ++i)
        _layers.emplace_back(_topology[i], _topology[i+1]);
    for (int i = 0; i < _topology.size(); ++i) {
        Vector temp = _layers[i].getActivations();
        Vector weigth = _layers[i].getWeight(i);
    }
}

NeuralNetwork::~NeuralNetwork() = default;

void NeuralNetwork::forwardPropagation() {
    for (int i = 1; i < _topology.size(); ++i) {
        _layers[i].forward(_layers[i-1].getActivations());
    }
    _out_layer = _layers.back().getActivations().empty()
        ? Vector(_topology.back(), 0.0)
        : _layers.back().getActivations();
}

void NeuralNetwork::backPropagation(const double lr) {
    Trainer compute(lr);
    compute.output(_layers.back(), _expected);
    compute.gradientDescent(_layers.back(), (_layers.rbegin() + 1)->getActivations());
    for (auto i = _layers.rbegin() + 1; i != _layers.rend() - 1; ++i) {
        compute.backward(*(i - 1), i->getActivations());
        compute.gradientDescent(*i, (i+1)->getActivations());
    }
}

double NeuralNetwork::meanSquareErr() {
    const size_t Size = _out_layer.size();
    double mse {0};
    for (int i = 0; i < Size; ++i) {
        mse += (_expected[i] - _out_layer[i]) * (_expected[i] - _out_layer[i]);
    }

    return mse / static_cast<double>(Size);
}

void NeuralNetwork::changeInput(const vector<double> &input,  const Vector &expected) {
    _layers[0] = Layer(_topology[0], _topology[1], input);
    _expected = expected;
}

std::string NeuralNetwork::output(const vector<std::string> &categories) const {
    int predicted = 0;
    for (int i = 1; i < static_cast<int>(categories.size()); ++i)
        if (_out_layer[predicted] < _out_layer[i])
            predicted = i;
    return categories[predicted];
}




