#include "headers/Neuron.h"

Neuron::Neuron(int n_inputs, double deviation) {
    std::mt19937_64 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, deviation);

    this->bias = 0.0;
    weights.reserve(n_inputs);

    for (int i = 0; i < n_inputs; i++)
        weights.push_back(dist(rng));
}

Neuron::Neuron(const std::vector<double>& values) : weights(values) {
    for (double weight : weights)
        if (!std::isfinite(weight))
            throw std::invalid_argument("Invalid weight value");

    this->bias = 0.0;
}

double Neuron::weighted_sum(const std::vector<double>& inputs) {
    if (weights.size() != inputs.size())
        throw std::invalid_argument("The size of the inputs must match the size of the weights");

    double sum = bias;
    for (size_t i = 0; i < weights.size(); i++)
        sum += weights[i] * inputs[i];

    z = sum;
    return sum;
}

std::vector<double>& Neuron::get_weights() {
    return weights;
}

double Neuron::get_z() {
    return z;
}

double Neuron::get_a() {
    return a;
}

void Neuron::set_a(double value) {
    a = value;
}

double Neuron::get_bias() {
    return bias;
}

void Neuron::set_bias(double value) {
    bias = value;
}