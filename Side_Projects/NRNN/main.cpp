#include "headers/Neuron.h"
#include "headers/Layer.h"
#include "headers/Network.h"
#include "headers/MNIST.h"

int main() {
    MNIST mnist;

    mnist.load(
        "mnist/train-images.idx3-ubyte",
        "mnist/train-labels.idx1-ubyte"
    );

    mnist.save_image(mnist.get_data()[0], ".out/image.txt");
}