#include "headers/Network.h"

Network::Network(int n_inputs, const std::vector<Layer_Architecture>& architectures, Loss_Type loss_type) 
: loss_type(loss_type) {
    if (architectures.size() < 2)
        throw std::invalid_argument("Network must have at least an input and output layer");

    layers.reserve(architectures.size() - 1);
    layers.emplace_back(architectures[0].n_neurons, n_inputs, architectures[0].activation_type);
    
    for (size_t i = 1; i < architectures.size(); i++)
        layers.emplace_back(architectures[i].n_neurons, 
            architectures[i-1].n_neurons, architectures[i].activation_type);
}

Network::Network(const std::vector<Layer>& layers, Loss_Type loss_type) 
: layers(layers), loss_type(loss_type) {}

std::vector<double> Network::forward_propagation(const std::vector<double>& inputs) {
    std::vector<double> outputs = inputs;
    this->inputs = inputs;

    for (Layer& layer : layers)
        outputs = layer.forward(outputs);
    return outputs;
}

double Network::loss(double y, double y_hat, Loss_Type type) {
    switch (type) {
        case BINARY_CROSS_ENTROPY:
            return binary_cross_entropy(y, y_hat);
        case CATEGORICAL_CROSS_ENTROPY:
            return categorical_cross_entropy(y, y_hat);
        case SSE:
            return sse(y, y_hat);
        default:
            throw std::invalid_argument("Invalid type");
    }
}

double Network::d_loss(double y, double y_hat, Loss_Type type) {
    switch (type) {
        case BINARY_CROSS_ENTROPY:
            return d_binary_cross_entropy(y, y_hat);
        case CATEGORICAL_CROSS_ENTROPY:
            return d_categorical_cross_entropy(y, y_hat);
        case SSE:
            return d_sse(y, y_hat);
        default:
            throw std::invalid_argument("Invalid type");
    }
}

std::vector<double> Network::d_loss_d_w(int L, int J, const std::vector<double>& Y) {
    if (L < 0 || static_cast<size_t>(L) >= layers.size()) 
        throw std::invalid_argument("Invalid L");

    Layer& current_layer = layers[L];
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

    for (size_t i = 0; i < current_weights.size(); i++) {
        double sum = 0.0;
        for (size_t j = 0; j < current_neurons.size(); j++)
            sum += current_layer.d_activate(j, J, {}, current_layer.get_activation_type()) * d_loss_d_a(L, j, Y);

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

    if (L == static_cast<int>(layers.size()) - 1)
        return d_loss(Y[J], current_neurons[J].get_a(), loss_type);

    Layer& next_layer = layers[L + 1];
    std::vector<Neuron>& next_neurons = next_layer.get_neurons();

    double result = 0.0;
    for (size_t i = 0; i < next_neurons.size(); i++) {
        double next_weight = next_neurons[i].get_weights()[J];
        
        double local_sum = 0.0;
        for (size_t j = 0; j < next_neurons.size(); j++)
            local_sum += next_layer.d_activate(j, i, {}, next_layer.get_activation_type()) * d_loss_d_a(L + 1, j, Y);

        result += (next_weight * local_sum);
    }

    return result;
}

void Network::update_weights(int L, int J, double learning_rate, const std::vector<double>& Y) {
    std::vector<double>& weights = layers[L].get_neurons()[J].get_weights();
    std::vector<double> grads = d_loss_d_w(L, J, Y);

    for (size_t i = 0; i < weights.size(); i++)
        weights[i] -= learning_rate * grads[i];
}

double Network::binary_cross_entropy(double y, double y_hat) {
    return -(y * std::log(y_hat) + (1 - y) * std::log(1 - y_hat));
}

double Network::d_binary_cross_entropy(double y, double y_hat) {
    return -((y / y_hat) - (1 - y) / (1 - y_hat));
}

double Network::sse(double y, double y_hat) {
    return (y - y_hat) * (y - y_hat);
}

double Network::d_sse(double y, double y_hat) {
    return -2 * (y - y_hat);
}

std::vector<Layer>& Network::get_layers() {
    return layers;
}

double Network::categorical_cross_entropy(double y, double y_hat) {
    return -y * std::log(y_hat);
}

double Network::d_categorical_cross_entropy(double y, double y_hat) {
    return -y / y_hat;
}

double Network::total_loss(const std::vector<double>& X, const std::vector<double>& Y) {
    std::vector<double> predictions = forward_propagation(X);
    double losses = 0.0;

    for (size_t i = 0; i < Y.size(); i++)
        losses += loss(Y[i], predictions[i], loss_type);

    return losses;
}