#include "headers/Layer.h"

Layer::Layer(int n_neurons, int n_inputs, Activation_Type activation_type) : activation_type(activation_type) {
    if (n_neurons <= 0)
        throw std::invalid_argument("Each layer must have at least one neuron");
    if (n_inputs <= 0)
        throw std::invalid_argument("Each neuron must take at least one input");

    double deviation = normal_Xavier_Deviation(n_inputs, n_neurons);

    neurons.reserve(n_neurons);
    for (int i = 0; i < n_neurons; i++)
        neurons.emplace_back(n_inputs, deviation);

}

Layer::Layer(const std::vector<std::vector<double>>& weights_layer, Activation_Type activation_type) 
: activation_type(activation_type) {
    neurons.reserve(weights_layer.size());

    if (!weights_layer.empty()) {
        size_t n_inputs = weights_layer[0].size();

        for (const auto& weights : weights_layer) {
            if (weights.size() != n_inputs)
                throw std::invalid_argument(
                    "All neurons in a layer must have the same number of weights"
                );
            neurons.emplace_back(weights);
        }
    } else {
        throw std::invalid_argument("Each neuron must have at least one weight");
    }
}

double Layer::normal_Xavier_Deviation(int n_inputs, int n_neurons) {
    return std::sqrt(2.0 / (n_inputs + n_neurons));
}

double Layer::relu(double x) {
    return std::max(0.0, x);
}

double Layer::softmax(double x, double eX, double mX) {
    return std::exp(x - mX) / eX;
}

double Layer::d_relu(double x) {
    return (x < 0.0) ? 0.0 : 1.0;
}

double Layer::d_softmax(int i, int j) {
    if (i == j) {
        double s = neurons[i].get_a();
        return s * (1 - s);
    } else {
        return - neurons[i].get_a() * neurons[j].get_a();
    }
}

double Layer::activate(const std::vector<double>& args, Activation_Type activation_type) {
    switch (activation_type) {
        case SOFTMAX:
            return softmax(args[0], args[1], args[2]);
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

double Layer::activate(double x, Activation_Type activation_type) {
    switch (activation_type) {
        case RELU:
            return relu(x);
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

double Layer::d_activate(int i, int j, const std::vector<double>& additional_args, Activation_Type activation_type) {
    switch (activation_type) {
        case SOFTMAX:
            return d_softmax(i, j);
        case RELU:
            if (i == j)
                return d_relu(neurons[i].get_z());
            else 
                return 0;
        default:
            throw std::invalid_argument("Unknown activation type");
    }
}

std::vector<double> Layer::weighted_sums(const std::vector<double>& inputs) {
    std::vector<double> sum;
    sum.reserve(neurons.size());
    for (Neuron& neuron : neurons) {
        double ws = neuron.weighted_sum(inputs);
        sum.push_back(ws);
    }

    return sum;
}

std::vector<double> Layer::forward(const std::vector<double>& inputs) {
    std::vector<double> sums = weighted_sums(inputs);

    std::vector<double> outputs;
    outputs.reserve(sums.size());

    switch (activation_type) {

        case SOFTMAX: {
            double max_sum = *std::max_element(sums.begin(), sums.end());
            double eX = 0.0;

            for (double sum : sums)
                eX += std::exp(sum - max_sum);

            for (size_t i = 0; i < sums.size(); ++i) {
                double a = activate({sums[i], eX, max_sum}, SOFTMAX);

                neurons[i].set_a(a);
                outputs.push_back(a);
            }

            break;
        }

        case RELU: {
            for (size_t i = 0; i < sums.size(); ++i) {
                double a = activate(sums[i], RELU);

                neurons[i].set_a(a);
                outputs.push_back(a);
            }

            break;
        }

        default:
            throw std::invalid_argument("Unknown activation type");
    }

    return outputs;
}

std::vector<Neuron>& Layer::get_neurons() {
    return neurons;
}

Activation_Type Layer::get_activation_type() {
    return activation_type;
}