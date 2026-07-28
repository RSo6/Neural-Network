#ifndef LAYER_H
#define LAYER_H
#include <vector>

using Vector = std::vector<double>;

class Layer {
    class Matrix {
        size_t _row {}, _col {}; // neuron & its weights
        std::vector<Vector> _data {};
        /*           w11 w12 ..  w1x128
         * neuron1  |x1  x2  ..  x128  | out_neuron1
         *           w21  w22 ..  w2x128
         * neuron2  |x2
         *          |..
         * neuron1000|x1000
         */
        void xavierNormal();

    public:
        Matrix();

        /**
         * @brief Constructs matrix of weights
         * @param in_layer Number of input layer neurons
         * @param out_layer Number of output layer neurons
         */
        Matrix(const size_t in_layer, const size_t out_layer) ;
        size_t rows() const;
        size_t cols() const;
        void updateWeight(int row, int col, double gradient);
        Vector column(int neuron) const; // 1000 x 1
        Vector row(int neuron) const; // 1 x 128
        // Matrix & operator=(const Matrix & weights);
    };
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