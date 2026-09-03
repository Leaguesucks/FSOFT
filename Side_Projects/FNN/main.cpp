#include <iostream>
#include <random>
#include <vector>
#include <algorithm>

#include "headers/Neuron.h"
#include "headers/Layer.h"
#include "headers/Network.h"
#include "headers/MNIST.h"
#include "headers/Adam.h"
#include "headers/test.h"

#define N_INPUTS 784
#define N_OUTPUTS 10
#define EPOCH_SIZE 20
#define BATCH_SIZE 256
#define SAVED_FILE "training/mnist_train.bin"

void train(std::vector<MNIST_Image>& training_data, const std::vector<MNIST_Image>& test_data, 
    Network& network, Adam& optimizer, std::mt19937& rng) {
    std::vector<double> Y(N_OUTPUTS, 0.0);

    std::cout << "Start training..." << "\n\n";
    double best_accuracy = 98.12; // The best accuracy so far
    double minimum_loss = 0.129846; // The minimum loss so far
    for (size_t epoch = 0; epoch < EPOCH_SIZE; epoch++) {
        std::shuffle(training_data.begin(), training_data.end(), rng);

        for (size_t start = 0; start < training_data.size(); start += BATCH_SIZE) {
            size_t end = std::min(start + BATCH_SIZE, training_data.size());
            size_t batch_size = end - start;
            optimizer.zero_gradients();

            for (size_t i = start; i < end; ++i) {
                const auto& image = training_data[i];

                std::fill(Y.begin(), Y.end(), 0.0);
                Y[image.label] = 1.0;
                optimizer.accumulate_gradients(image.pixels, Y);
            }
            optimizer.update_weights(batch_size);
        }
        
        // Evaluate the model on the test dataset
        double total_loss = 0.0;
        size_t n_correct = 0;
        for (size_t i = 0; i < test_data.size(); i++) {
            const auto& image = test_data[i];
            std::fill(Y.begin(), Y.end(), 0.0);
            Y[image.label] = 1.0;

            total_loss += network.total_loss(image.pixels, Y); // Call forward_propagation
            const auto& Y_HAT = network.get_Y_HAT();
            
            size_t predicted_label = std::distance(Y_HAT.begin(), std::max_element(Y_HAT.begin(), Y_HAT.end()));
            if (predicted_label == image.label)
                n_correct++;
        }

        double accuracy = static_cast<double>(n_correct) / test_data.size() * 100.0;
        double average_loss = total_loss / test_data.size();

        if (accuracy > best_accuracy) {
            std::cout << "CURRENT BEST\n";
            best_accuracy = accuracy;
            network.save(SAVED_FILE);
        } else if (accuracy == best_accuracy && average_loss < minimum_loss) {
            std::cout << "CURRENT BEST\n";
            minimum_loss = average_loss;
            network.save(SAVED_FILE);
        }

        std::cout << "Epoch:        " << epoch + 1 <<"\n";
        std::cout << "Average loss: " << average_loss << "\n";
        std::cout << "Accuracy:     " << accuracy << "%\n\n";
    }
}

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

    // Network network(N_INPUTS, 
    //     {{128, RELU}, {64, RELU}, {10, SOFTMAX}},
    //     CATEGORICAL_CROSS_ENTROPY
    // );
    Network network(SAVED_FILE);
    network.set_accumulate_gradients(true);

    Adam optimizer(network);

    train(mnist.get_data(), mnist_test.get_data(), network, optimizer, g);
    // network.save(SAVED_FILE);

    // for (size_t i = 0; i < 20; i++) { // For the displayer to read
    //     mnist_test.save_image(mnist_test.get_data()[i], ".out/test_" + std::to_string(i) + ".out");
    // }

    // Network network = Network(N_INPUTS, 
    //     {{128, RELU}, {64, RELU}, {10, SOFTMAX}},
    //     CATEGORICAL_CROSS_ENTROPY);

    // network.load(SAVED_FILE);

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




    // const auto& test_data = mnist_test.get_data();
    // for (size_t i = 0; i < 20; ++i) {
    //     const auto& image = test_data[i];

    //     network.forward_propagation(image.pixels);

    //     const auto& prediction = network.get_Y_HAT();
    //     size_t predicted_label =
    //         std::distance(
    //             prediction.begin(),
    //             std::max_element(
    //                 prediction.begin(),
    //                 prediction.end()
    //             )
    //         );

    //     std::cout << "========================================\n";
    //     std::cout << "Test image:      " << i << '\n';
    //     std::cout << "Actual label:    "
    //             << static_cast<int>(image.label)
    //             << '\n';

    //     std::cout << "Predicted label: "
    //             << predicted_label
    //             << '\n';

    //     std::cout << "Probabilities:\n";

    //     for (size_t j = 0; j < prediction.size(); ++j) {
    //         std::cout
    //             << "  ["
    //             << j
    //             << "] "
    //             << prediction[j] * 100.0
    //             << "%\n";
    //     }

    //     std::cout << '\n';
    // }

    return 0;
}