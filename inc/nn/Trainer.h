#ifndef TRAINER_H
#define TRAINER_H

#include "Layer.h"
class Trainer {
    double _learning_rate {};
    size_t _deltaSize {};
    size_t _gradientSize {};
    Vector _post_delta {};
    Vector _delta {};
public:
    explicit Trainer(double lr = 0.01);
    void backward(Layer &layer,  const Vector &previous, const Vector &expected);
    void gradientDescent(Layer &layer, const Vector &previous) const; // w = w - lr(gradient);


};

#endif //TRAINER_H
