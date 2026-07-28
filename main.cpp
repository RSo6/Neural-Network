#include <iostream>
#include "inc/nn/NeuralNetwork.h"

int main() {
    using std::cout;
    using std::cin;
    using std::endl;

    vector<std::string> categories {
        "University",
        "Work",
        "Finance",
        "Programming",
        "Personal",
        "Media",
        "Books",
        "System",
        "Archives",
        "Misc"
    };

    NeuralNetwork nn({3, 128, 64, 10}, {0.5, 2.4, 0.91}, {0,1,0,0,0,0,0,0,0,0});
    for (int i = 0; i < 1000000; ++i) {
        nn.forwardPropagation();
        cout << nn.output(categories) << " -> " << nn.meanSquareErr() << " cost." << endl;
        nn.backPropagation(1.1);
        cout << nn.output(categories) << " -> " << nn.meanSquareErr() << " cost." << endl;
    }

    return 0;
}
