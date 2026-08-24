#include "headers/Layer.h"

Layer::Layer(int n_neurons, int n_inputs, int type) : type(type) {
    if (type < 0 || type > 2)
        throw std::invalid_argument("Invalid type");

    double deviation = normal_Xavier_Deviation(n_inputs, n_neurons);

    neurons.reserve(n_neurons);
    for (int i = 0; i < n_neurons; i++)
        neurons.emplace_back(n_inputs, deviation);

}

Layer::Layer(const std::vector<std::vector<double>>& weights_layer, int type) : type(type) {
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

double Layer::d_softmax(double x, double eX, double mX) {
    double s = softmax(x, eX, mX);
    return s * (1.0 - s);
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
    double max_sum, eX;
    std::vector<double> sums = weighted_sums(inputs);
    std::vector<double> outputs;
    outputs.reserve(sums.size());

    if (type == OUTPUT_LAYER) {
        max_sum = *std::max_element(sums.begin(), sums.end());
        eX = 0.0;
        for (double sum : sums)
            eX += std::exp(sum - max_sum);
        for (double sum : sums)
            outputs.push_back(softmax(sum, eX, max_sum));
    } else {
        for (double sum : sums)
            outputs.push_back(relu(sum));
    }

    return outputs;
}
