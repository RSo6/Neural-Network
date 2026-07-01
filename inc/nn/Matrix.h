#ifndef MATRIX_H
#define MATRIX_H
#include <vector>

using std::vector;
class Matrix {
    size_t _row {}, _col {}; // neuron & its weights
    vector<vector<double>> _data {};
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
    Matrix(const size_t in_layer, const size_t out_layer) ;
    size_t rows() const;
    size_t cols() const;
    void updateWeight(int row, int col, double gradient);
    vector<double> column(int neuron) const; // 1000 x 1
    vector<double> row(int neuron) const; // 1 x 128
    // Matrix & operator=(const Matrix & weights);
};
#endif //MATRIX_H
