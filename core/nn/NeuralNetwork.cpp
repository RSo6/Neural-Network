#include "../../inc/nn/NeuralNetwork.h"
#include "../../inc/nn/Trainer.h"


NeuralNetwork::NeuralNetwork() : NeuralNetwork(
    {1000, 128, 64, 10}, {}, {}) {}

NeuralNetwork::NeuralNetwork(const vector<int> &topology, const Vector &input, const Vector &expected)
        : _topology(topology), _expected(expected) {
    _layers.emplace_back(_topology[0], _topology[1], input);
    for (int i = 1; i < _topology.size() - 1; ++i)
        _layers.emplace_back(_topology[i], _topology[i+1]);
}

NeuralNetwork::~NeuralNetwork() = default;

void NeuralNetwork::forwardPropagation() {
    for (int i = 1; i < _topology.size(); ++i)
        _layers[i].forward(_layers[i-1].getActivations());
    if (!_layers.back().getActivations().empty())
        _out_layer = _layers.back().getActivations();
    else
        _out_layer = {0.0};
}

void NeuralNetwork::backPropagation(const double lr) {
    Trainer compute(lr);
    compute.backward(_layers.back(), _out_layer, _expected);
    for (auto i = _layers.rbegin(); i != _layers.rend() - 1; ++i) {
        compute.backward(*i, (i+1)->getActivations(),_expected);
        compute.gradientDescent(*i, (i+1)->getActivations());
    }
}

double NeuralNetwork::meanSquareErr() {
    const size_t Size = _out_layer.size();
    // _squared_diff.assign(Size, 0.0);
    double mse {};
    for (int i = 0; i < Size; ++i)
        mse += (_expected[i] - _out_layer[i]) * (_expected[i] - _out_layer[i]);
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




