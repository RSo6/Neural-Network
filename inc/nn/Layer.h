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
    /**
     * @brief Constructs network's layer
     * @param in Number of current layer neurons
     * @param out Number of toward layer neurons
     * @param input Values for input layer, used only for 2nd layer values calculation
     */
    Layer(size_t in, size_t out, const Vector &input = {0.0});

    /**
     * @brief Feed forward per layer algorithm
     * @param input Values of current layer, based on which forward-propagation calculates
     *              next layer values
     */
    void forward(Vector &input);

    /**
     * @brief Update the layer's weights, part of back-propagation
     * @param row Matrix's row position of weight
     * @param col Matrix's column position of weight
     * @param gradient Value to update in matrix of weights
     */
    void updateWeight(int row, int col, double gradient);

    /**
     * @param neuron Layer's neuron used to get its weights
     * @return Vector of weights connected to neuron
     */
    Vector getWeight(int neuron) const;

    /**
     * @return Vector of current layer's activations
     */
    Vector & getActivations();
    size_t previousLayerSize() const;
    static Vector toSigmoid(const Vector &logits);

    /**
     * @brief Runs multiplication of current layer's values by weights,
     *        dot product calculation
     * @param activations Set of current layer's values
     * @param weights Set of weights between current and forward layers
     * @return Vector result of dot product operation
     */
    friend Vector operator*(Vector &activations, const Matrix &weights);

    /**
     * @brief Runs summation of biases to calculated dot product Vector to prepare
     *        the values for activation function
     * @param product Vector of dot product operation
     * @param biases Vector of biases
     * @return Vector of pre-activation output
     */
    friend Vector operator+(const Vector &product, const Vector &biases);
};

#endif //LAYER_H