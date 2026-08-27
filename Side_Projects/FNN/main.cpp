#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#include "headers/Neuron.h"
#include "headers/Layer.h"
#include "headers/Network.h"
#include "headers/MNIST.h"
#include "headers/Adam.h"

#define N_INPUTS 784
#define N_OUTPUTS 10
#define EPOCH_SIZE 200
#define BATCH_SIZE 500
#define SAVED_FILE "training/mnist_train.bin"

int main() {
    std::random_device rd;
    std::mt19937 g(rd());

    MNIST mnist, mnist_test;

    mnist.load(
        "mnist/train-images.idx3-ubyte",
        "mnist/train-labels.idx1-ubyte"
    );

    mnist_test.load(
        "mnist/t10k-images.idx3-ubyte",
        "mnist/t10k-labels.idx1-ubyte"
    );

    // for (size_t i = 0; i < 20; i++) { // For the displayer to read
    //     mnist_test.save_image(mnist_test.get_data()[i], ".out/test_" + std::to_string(i) + ".out");
    // }

    Network network = Network(N_INPUTS, 
        {{128, RELU}, {64, RELU}, {10, SOFTMAX}},
        CATEGORICAL_CROSS_ENTROPY);

    network.load(SAVED_FILE);

    // Adam optimizer = Adam(network);

    // auto& training_dataS = mnist.get_data();
    // std::vector<double> Y(N_OUTPUTS, 0.0);

    // std::shuffle(training_dataS.begin(), training_dataS.end(), g);
    // std::vector<MNIST_Image> samples;
    // samples.reserve(BATCH_SIZE);

    // for (size_t i = 0; i < BATCH_SIZE; i++)
    //     samples.push_back(training_dataS[i]);

    // std::cout << "Start training..." << "\n";
    // for (size_t epoch = 0; epoch < EPOCH_SIZE; epoch++) {
    //     std::shuffle(training_dataS.begin(), training_dataS.end(), g);

    //     for (size_t i = 0; i < BATCH_SIZE; i++) {
    //         const auto& image = training_dataS[i];
    //         std::fill(Y.begin(), Y.end(), 0.0);
    //         Y[image.label] = 1.0;
            
    //         optimizer.update_weights(image.pixels, Y);
    //     }
        
    //     // Evaluate the training
    //     double total_loss = 0.0;
    //     size_t n_correct = 0.0;
    //     for (size_t i = 0; i < BATCH_SIZE; i++) {
    //         const auto& image = samples[i];
    //         std::fill(Y.begin(), Y.end(), 0.0);
    //         Y[image.label] = 1.0;

    //         total_loss += network.total_loss(image.pixels, Y); // Call forward_propagation
    //         const auto& Y_HAT = network.get_Y_HAT();
            
    //         size_t predicted_label = std::distance(Y_HAT.begin(), std::max_element(Y_HAT.begin(), Y_HAT.end()));
    //         if (predicted_label == image.label)
    //             n_correct++;
    //     }

    //     std::cout << "Epoch:        " << epoch <<"\n";
    //     std::cout << "Average loss: " << total_loss / BATCH_SIZE << "\n";
    //     std::cout << "Accuracy:     " << static_cast<double>(n_correct) / BATCH_SIZE * 100.0 << "%\n\n";
    // }
    
    // network.save(SAVED_FILE);
    // std::cout << "File saved successfully\n";




    const auto& test_data = mnist_test.get_data();
    for (size_t i = 0; i < 20; ++i) {
        const auto& image = test_data[i];

        network.forward_propagation(image.pixels);

        const auto& prediction = network.get_Y_HAT();
        size_t predicted_label =
            std::distance(
                prediction.begin(),
                std::max_element(
                    prediction.begin(),
                    prediction.end()
                )
            );

        std::cout << "========================================\n";
        std::cout << "Test image:      " << i << '\n';
        std::cout << "Actual label:    "
                << static_cast<int>(image.label)
                << '\n';

        std::cout << "Predicted label: "
                << predicted_label
                << '\n';

        std::cout << "Probabilities:\n";

        for (size_t j = 0; j < prediction.size(); ++j) {
            std::cout
                << "  ["
                << j
                << "] "
                << prediction[j] * 100.0
                << "%\n";
        }

        std::cout << '\n';
    }

    return 0;
}