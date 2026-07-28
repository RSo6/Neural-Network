#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include <string>
using std::vector;

class NeuralNetwork {
    vector<int> _topology {};
    vector<Layer> _layers;
    Vector _out_layer {};
    Vector _expected {};
public:
    /**
     * @brief Constructs neural network
     * @param topology Number of neurons per layer, ordered input->output
     * @param input Input layer values
     * @param expected Target output values used to compute errors against the neural network's output
     */
    NeuralNetwork(const vector<int> & topology, const Vector & input, const Vector & expected);
    NeuralNetwork();
    ~NeuralNetwork();

    /**
     * @brief Updates the network's input and expected output values
     * @param input Updated input layer values
     * @param expected Updated target output values for particular input
     */
    void changeInput(const Vector &input, const Vector &expected);

    /**
     * @brief Runs the feed forward propagation per layer algorithm,
     *        computing each layer's activations based on previous layer's output
     */
    void forwardPropagation();

    /**
     * @brief Runs the back propagation per layer algorithm which
     *        will update the weights and make network predictions more accurate
     * @param lr Learning rate for gradient descent. Must be > 0;
     *           larger values speed up training but risk divergence.
     */
    void backPropagation(double lr);

    /**
    * @brief Compute error against predicted output layer's values
    */
    double meanSquareErr();

    /**
     * @brief Display neural network's predicted value
     * @param categories // fix invoke of the array each time this function it called
     * @return Name of
     */
    std::string output(const vector<std::string> & categories) const;

};
#endif //NEURALNETWORK_H
