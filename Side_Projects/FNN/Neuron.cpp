#include "headers/Neuron.h"

Neuron::Neuron(int n_inputs, double deviation) :
bias(0.0), z(0.0), a(0.0), delta(0.0), dLda(0.0),
mts(n_inputs, 0.0), vts(n_inputs, 0.0),
bias_mt(0.0), bias_vt(0.0) {
    std::mt19937_64 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, deviation);

    weights.reserve(n_inputs);

    for (int i = 0; i < n_inputs; i++) {
        weights.push_back(dist(rng));
    }
}

Neuron::Neuron(const std::vector<double>& values) : 
weights(values), bias(0.0), z(0.0), a(0.0), delta(0.0), dLda(0.0),
mts(values.size(), 0.0), vts(values.size(), 0.0),
bias_mt(0.0), bias_vt(0.0) {

    for (double weight : weights) {
        if (!std::isfinite(weight))
            throw std::invalid_argument("Invalid weight value");
    }
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

std::vector<double>& Neuron::get_gradients() {
    return gradients;
}

std::vector<double>& Neuron::get_mts() {
    return mts;
}

std::vector<double>& Neuron::get_vts() {
    return vts;
}

double Neuron::get_delta() {
    return delta;
}

void Neuron::set_delta(double value) {
    delta = value;
}

double Neuron::get_dLda() {
    return dLda;
}

void Neuron::set_dLda(double value) {
    dLda = value;
}

double Neuron::get_bias_mt() {
    return bias_mt;
}

void Neuron::set_bias_mt(double value) {
    bias_mt = value;
}

double Neuron::get_bias_vt() {
    return bias_vt;
}

void Neuron::set_bias_vt(double value) {
    bias_vt = value;
}