#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include <string>
#include <vector>
using std::vector;

class NeuralNetwork {
    vector<int> _topology {};
    vector<Layer> _layers;
    Vector _out_layer {};
    Vector _expected {};
    // Vector _squared_diff{};
public:
    NeuralNetwork(const vector<int> & topology, const Vector & input, const Vector & expected);
    NeuralNetwork();
    ~NeuralNetwork();
    void changeInput(const Vector &input, const Vector &expected);
    void forwardPropagation();
    void backPropagation(double lr);
    double meanSquareErr();
    std::string output(const vector<std::string> & categories) const;

};
#endif //NEURALNETWORK_H
