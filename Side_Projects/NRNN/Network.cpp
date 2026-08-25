#include "headers/Network.h"

Network::Network(const std::vector<int>& architecture) {
    if (architecture.size() < 2)
        throw std::invalid_argument("Network must have at least an input and output layer");

    layers.reserve(architecture.size() - 1);

    for (size_t i = 1; i < architecture.size(); i++) {
        int n_inputs = architecture[i-1];
        int n_neurons = architecture[i];
        int type;

        inputs.reserve(n_inputs);
        if (i == architecture.size() - 1)
            type = OUTPUT_LAYER;
        else
            type = HIDDEN_LAYER;

        layers.emplace_back(n_neurons, n_inputs, type);
    }
}

Network::Network(const std::vector<Layer>& layers) : layers(layers) {}

std::vector<double> Network::forward_propagation(const std::vector<double>& inputs) {
    std::vector<double> outputs;
    outputs.reserve(inputs.size());

    this->inputs = inputs;
    std::copy(inputs.begin(), inputs.end(), std::back_inserter(outputs));

    for (Layer& layer : layers)
        outputs = layer.forward(outputs);
    return outputs;
}

double Network::loss(double y, double y_hat) {
    return -(y * std::log(y_hat) + (1 - y) * std::log(1 - y_hat)); // Binary Cross-Entropy
}

double Network::d_loss(double y, double y_hat) {
    return -((y / y_hat) - (1 - y) / (1 - y_hat));
}

std::vector<double> Network::d_loss_d_w(int L, int J, const std::vector<double>& Y) {
    if (L < 0 || static_cast<size_t>(L) >= layers.size()) 
        throw std::invalid_argument("Invalid L");

    Layer current_layer = layers[L];
    std::vector<Neuron>& current_neurons = current_layer.get_neurons();
    if (J < 0 || static_cast<size_t>(J) >= current_neurons.size())
        throw std::invalid_argument("Invalid J");

    std::vector<double> current_inputs;
    if (L == 0) {
        current_inputs = inputs;
    } else {
        std::vector<Neuron>& prev_neurons = layers[L - 1].get_neurons();
        current_inputs.reserve(prev_neurons.size());

        for (Neuron& neuron : prev_neurons)
            current_inputs.push_back(neuron.get_a());
    }

    Neuron& current_neuron = current_neurons[J];
    std::vector<double>& current_weights = current_neuron.get_weights();

    std::vector<double> results;
    results.reserve(current_weights.size());

    for (int i = 0; i < current_weights.size(); i++) {
        double sum = 0.0;
        for (int j = 0; j < current_neurons.size(); j++)
            if (current_layer.get_type() == OUTPUT_LAYER)
                sum += current_layer.d_activate(j, J, current_layer.get_additional_data(), SOFTMAX) * d_loss_d_a(L, j, Y);
            else
                sum += current_layer.d_activate(j, j, {}, RELU) * d_loss_d_a(L, j, Y);

        results.push_back(current_inputs[i] * sum);
    }

    return results;
}

double Network::d_loss_d_a(int L, int J, const std::vector<double>& Y) {
    if (L < 0 || static_cast<size_t>(L) >= layers.size())
        throw std::invalid_argument("Invalid L");

    Layer& current_layer = layers[L];
    std::vector<Neuron>& current_neurons = current_layer.get_neurons();

    if (J < 0 || static_cast<size_t>(J) >= current_neurons.size())
        throw std::invalid_argument("Invalid J");

    if (current_layer.get_type() == OUTPUT_LAYER)
        return d_loss(Y[J], current_neurons[J].get_a());

    Layer& next_layer = layers[L + 1];
    std::vector<Neuron>& next_neurons = next_layer.get_neurons();

    double result = 0.0;
    for (int i = 0; i < next_neurons.size(); i++) {
        double next_weight = next_neurons[i].get_weights()[J];
        
        double local_sum = 0.0;
        for (int j = 0; j < next_neurons.size(); j++) {
            if (next_layer.get_type() == OUTPUT_LAYER)
                local_sum += next_layer.d_activate(j, i, next_layer.get_additional_data(), SOFTMAX) * d_loss_d_a(L + 1, i, Y);
            else
                local_sum += next_layer.d_activate(j, i, {}, RELU) * d_loss_d_a(L + 1, i, Y);
        }

        result += (next_weight * local_sum);
    }

    return result;
}