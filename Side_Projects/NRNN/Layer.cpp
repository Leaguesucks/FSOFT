#include "headers/Layer.h"

Layer::Layer(int n_neurons, int n_inputs, Layer_Type type) : type(type) {
    if (type != HIDDEN_LAYER && type != OUTPUT_LAYER)
        throw std::invalid_argument("Invalid type");

    double deviation = normal_Xavier_Deviation(n_inputs, n_neurons);

    neurons.reserve(n_neurons);
    for (int i = 0; i < n_neurons; i++)
        neurons.emplace_back(n_inputs, deviation);

}

Layer::Layer(const std::vector<std::vector<double>>& weights_layer, Layer_Type type) : type(type) {
    if (type < 0 || type > 2)
        throw std::invalid_argument("Invalid type");

    neurons.reserve(weights_layer.size());
    for (const std::vector<double>& weights : weights_layer)
        neurons.emplace_back(weights);
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
            return d_relu(neurons[i].get_z());
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

    if (type == OUTPUT_LAYER) {
        double max_sum = *std::max_element(sums.begin(), sums.end());
        double eX = 0.0;

        for (double sum : sums)
            eX += std::exp(sum - max_sum);

        additional_data = {eX, max_sum};

        for (int i = 0; i < sums.size(); i++) {
            double sum = sums[i];
            double a = activate({sum, eX, max_sum}, SOFTMAX);
            neurons[i].set_a(a);
            neurons[i].set_da_dz(d_activate(i, i ,{eX, max_sum}, SOFTMAX)); // For back propagation
            outputs.push_back(a);
        }
    } else {
        for (int i = 0; i < sums.size(); i++) {
            double sum = sums[i];
            double a = activate(sum, RELU);

            neurons[i].set_a(a);
            neurons[i].set_da_dz(d_activate(i, i, {neurons[i].get_z()}, RELU)); // For back propagation
            outputs.push_back(a);
        }
    }

    return outputs;
}

std::vector<Neuron>& Layer::get_neurons() {
    return neurons;
}

Layer_Type Layer::get_type() {
    return type;
}

std::vector<double>& Layer::get_additional_data() {
    return additional_data;
}