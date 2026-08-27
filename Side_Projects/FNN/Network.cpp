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

void Network::forward_propagation(const std::vector<double>& inputs) {
    Y_HAT = inputs;
    this->X = inputs;

    for (Layer& layer : layers)
        Y_HAT = layer.forward(Y_HAT);
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
    return -y / (y_hat + 1e-9);
}

double Network::total_loss(const std::vector<double>& X, const std::vector<double>& Y) {
    double losses = 0.0;

    forward_propagation(X);
    for (size_t i = 0; i < Y.size(); i++)
        losses += loss(Y[i], Y_HAT[i], loss_type);

    return losses;
}

void Network::back_propagation(const std::vector<double>& Y) {
    for (size_t l = layers.size(); l-- > 0; ) {
        Layer& current_layer = layers[l];
        Activation_Type atype = current_layer.get_activation_type();
        auto& current_neurons = current_layer.get_neurons();

        // Calculate dLda for each neuron in this layer
        for (size_t k = 0; k < current_neurons.size(); k++) {
            Neuron& neuron = current_neurons[k];
    
            if (l == layers.size() - 1)
                neuron.set_dLda(d_loss(Y[k], neuron.get_a(), loss_type));
            else {
                Layer& next_layer = layers[l + 1];
                auto& next_neurons = next_layer.get_neurons();

                double sum = 0.0;
                for (size_t j = 0; j < next_neurons.size(); j++) {
                    double weight = next_neurons[j].get_weights()[k];
                    double delta = next_neurons[j].get_delta();
                    sum += weight * delta;
                }

                neuron.set_dLda(sum);
            }
        }

        // Calculate delta for each neuron in this layer
        for (size_t i = 0; i < current_neurons.size(); i++) {
            double delta = 0.0;
            for (size_t j = 0; j < current_neurons.size(); j++) {
                delta += current_neurons[j].get_dLda() * current_layer.d_activate(j, i, {}, atype);
            } 
            current_neurons[i].set_delta(delta);
        }

        // Calculate the gradient for each neuron in this layer
        for (size_t i = 0; i < current_neurons.size(); i++) {
            auto& gradients = current_neurons[i].get_gradients();
            auto& weights = current_neurons[i].get_weights();
            gradients.clear();
            gradients.reserve(weights.size());
            
            for (size_t j = 0; j < weights.size(); j++) {
                double a = (l == 0) ? X[j] : layers[l - 1].get_neurons()[j].get_a();
                gradients.push_back(current_neurons[i].get_delta() * a);
            }
        }
    }
}

void Network::forward_back_propagation(const std::vector<double>& X, const std::vector<double>& Y) {
    forward_propagation(X);
    back_propagation(Y);
}

std::vector<double>& Network::get_Y_HAT() {
    return Y_HAT;
}

void Network::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);

    if (!file)
        throw std::runtime_error("Could not open file for writing: " + filename);

    constexpr char MAGIC[] = "FNN";
    constexpr uint32_t VERSION = 1;

    file.write(MAGIC, sizeof(MAGIC) - 1);
    file.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));

    uint64_t input_size = X.size();
    file.write(reinterpret_cast<const char*>(&input_size), sizeof(input_size));

    uint64_t n_layers = layers.size();
    file.write(reinterpret_cast<const char*>(&n_layers), sizeof(n_layers));

    for (Layer& layer : layers) {
        auto& neurons = layer.get_neurons();
        uint64_t n_neurons = neurons.size();

        file.write(reinterpret_cast<const char*>(&n_neurons), sizeof(n_neurons));

        uint32_t activation = static_cast<uint32_t>(layer.get_activation_type());
        file.write(reinterpret_cast<const char*>(&activation), sizeof(activation));

        for (Neuron& neuron : neurons) {
            auto& weights = neuron.get_weights();
            uint64_t n_weights = weights.size();

            file.write(reinterpret_cast<const char*>(&n_weights), sizeof(n_weights));
            file.write(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(double));

            double bias = neuron.get_bias();
            file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
        }
    }

    if (!file)
        throw std::runtime_error("Error occured while saving the network");
}

void Network::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file)
        throw std::runtime_error("Could not open file: " + filename);

    char magic[3];
    file.read(magic, sizeof(magic));

    if (std::string(magic, 3) != "FNN")
        throw std::runtime_error("Invalid saved file");

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (version != 1)
        throw std::runtime_error("Unsupproted version");

    uint64_t input_size;
    file.read(reinterpret_cast<char*>(&input_size), sizeof(input_size));

    uint64_t n_layers;
    file.read(reinterpret_cast<char*>(&n_layers), sizeof(n_layers));

    layers.clear();
    layers.reserve(n_layers);

    for (uint64_t L = 0; L < n_layers; L++) {
        uint64_t n_neurons;
        file.read(reinterpret_cast<char*>(&n_neurons), sizeof(n_neurons));

        uint32_t activation_value;
        file.read(reinterpret_cast<char*>(&activation_value), sizeof(activation_value));

        Activation_Type activation = static_cast<Activation_Type>(activation_value);

        std::vector<std::vector<double>> weights;
        weights.reserve(n_neurons);

        std::vector<double> biases;
        biases.reserve(n_neurons);

        for (uint64_t J = 0; J < n_neurons; J++) {
            uint64_t n_weights;
            file.read(reinterpret_cast<char*>(&n_weights), sizeof(n_weights));

            std::vector<double> neuron_weights(n_weights);
            file.read(reinterpret_cast<char*>(neuron_weights.data()), n_weights * sizeof(double));

            double bias;
            file.read(reinterpret_cast<char*>(&bias), sizeof(bias));

            weights.push_back(std::move(neuron_weights));

            biases.push_back(bias);
        }

        Layer layer(weights, activation);
        auto& neurons = layer.get_neurons();

        for (size_t J = 0; J < neurons.size(); J++)
            neurons[J].set_bias(biases[J]);

        layers.push_back(std::move(layer));
    }

    if (!file)
        throw std::runtime_error("Error occured while reading from saved file");
    X.resize(input_size);
}