#include "headers/Adam.h"

Adam::Adam(Network& network,
           double a,
           double B1, double B2,
           double epsilon) :
           a(a), B1(B1), B2(B2),
           epsilon(epsilon), network(network),
           t(0) {
    if (a <= 0.0)
        throw std::invalid_argument("Learning rate must be greater than zero");

    if (B1 < 0.0 || B1 >= 1.0)
        throw std::invalid_argument("B1 must be in range [0, 1)");

    if (B2 < 0.0 || B2 >= 1.0)
        throw std::invalid_argument("B2 must be in range [0, 1)");

    if (epsilon <= 0.0)
        throw std::invalid_argument("Epsilon must be greater than zero");
}

void Adam::update_weights(const std::vector<double>& X, const std::vector<double>& Y) {
    network.forward_back_propagation(X, Y);
    ++t; // Advance Adam time step

    const double bias_correction_1 = 1.0 - std::pow(B1, static_cast<double>(t));
    const double bias_correction_2 = 1.0 - std::pow(B2, static_cast<double>(t));

    for (Layer& layer : network.get_layers()) {
        for (Neuron& neuron : layer.get_neurons()) {
            auto& weights = neuron.get_weights();
            auto& grads = neuron.get_gradients();
            auto& mts = neuron.get_mts();
            auto& vts = neuron.get_vts();

            if (weights.size() != grads.size())
                throw std::runtime_error("Weights and gradient sizes do not match");
            if (mts.size() != weights.size() || vts.size() != weights.size())
                throw std::runtime_error("Adam moment vectors must match weight vector size");

            // Update weights
            for (size_t i = 0; i < weights.size(); i++) {
                double mt_hat, vt_hat;
                double g = grads[i];

                mts[i] = B1 * mts[i] + (1.0 - B1) * g;
                vts[i] = B2 * vts[i] + (1.0 - B2) * g * g;

                mt_hat = mts[i] / bias_correction_1;
                vt_hat = vts[i] / bias_correction_2;

                weights[i] -= a * mt_hat / (std::sqrt(vt_hat) + epsilon);
            }
            
            // Update bias
            double bias_mt_hat, bias_vt_hat;
            double bias_mt = neuron.get_bias_mt();
            double bias_vt = neuron.get_bias_vt();
            double bias = neuron.get_bias();
            double bias_grad = neuron.get_delta();

            bias_mt = B1 * bias_mt + (1.0 - B1) * bias_grad;
            bias_vt = B2 * bias_vt + (1.0 - B2) * bias_grad * bias_grad;

            bias_mt_hat = bias_mt / bias_correction_1;
            bias_vt_hat = bias_vt / bias_correction_2;

            bias -= a * bias_mt_hat / (std::sqrt(bias_vt_hat) + epsilon);

            neuron.set_bias_mt(bias_mt);
            neuron.set_bias_vt(bias_vt);
            neuron.set_bias(bias);
        }
    }
}