#include "headers/Neuron.h"
#include "headers/Layer.h"
#include "headers/Network.h"
#include "headers/MNIST.h"

#include "headers/test.h"

#define N_INPUTS 784

int main() {
    // MNIST mnist;

    // mnist.load(
    //     "mnist/train-images.idx3-ubyte",
    //     "mnist/train-labels.idx1-ubyte"
    // );

    // mnist.save_image(mnist.get_data()[0], ".out/image.txt");
    
    Network network = Network(N_INPUTS, 
        {{128, RELU}, {64, RELU}, {10, SOFTMAX}},
        CATEGORICAL_CROSS_ENTROPY);

    return 0;
}