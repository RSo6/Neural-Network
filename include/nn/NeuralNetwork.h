#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include <string>
#include <vector>

class NeuralNetwork {
    const int NetSize {4};
    std::string _categories[] {
        "University", "Work", "Finance", "Programming", "Personal", "Media",
        "Books", "System", "Archives", "Misc"
    };
    std::vector<Layer> _layers;

public:
    NeuralNetwork(const Vector & input);
    NeuralNetwork();
    ~NeuralNetwork();
    void changeInput();
    void forwardPropagation();
    void backPropagation();
    std::string output() const;

};
#endif //NEURALNETWORK_H
