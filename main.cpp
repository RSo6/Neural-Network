#include <iostream>
#include "inc/nn/NeuralNetwork.h"

int main() {
    using std::cout;
    using std::cin;
    using std::endl;

    vector<std::string> categories {
        "University", "Work", "Finance", "Programming", "Personal", "Media",
        "Books", "System", "Archives", "Misc"
    };

    NeuralNetwork nn({3, 128, 64, 10}, {0.5, 2.4, 0.91}, {0,1,0,0,0,0,0,0,0,0});
    nn.forwardPropagation();
    cout << nn.output(categories) << " -> " << nn.meanSquareErr() << " cost." << endl;
    nn.backPropagation(0.01);
    cout << nn.output(categories) << " -> " << nn.meanSquareErr() << " cost." << endl;
    nn.changeInput({}, {});
    /*repeat*/

    return 0;
}
