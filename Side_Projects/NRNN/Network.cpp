#include "headers/Network.h"

Network::Network(const std::vector<int>& architecture) {
    if (architecture.size() < 2)
        throw std::invalid_argument("Network must have at least an input and output layer");

    layers.reserve(architecture.size() - 1);

    for (size_t i = 1; i < architecture.size(); i++) {
        int n_inputs = architecture[i-1];
        int n_neurons = architecture[i];
        int type;

        if (i == architecture.size() - 1)
            type == OUTPUT_LAYER;
        else
            type == HIDDEN_LAYER;

        layers.emplace_back(n_neurons, n_inputs, type);
    }
}

Network::Network(const std::vector<Layer>& layers) : layers(layers) {}

std::vector<double> Network::forward_propagation(std::vector<double> inputs) {
    for (Layer& layer : layers)
        inputs = layer.forward(inputs);
    return inputs;
}

double Network::loss(double y, double y_hat) {
    return -(y * std::log(y_hat) + (1 - y) * std::log(1 - y_hat)); // Binary Cross-Entropy
}

double Network::d_loss(double y, double y_hat) {
    return y_hat - y;
}