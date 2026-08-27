#pragma once

#include <random>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>

#define MAX_INPUTS 784
#define MIN_VAL 0
#define MAX_VAL 255

/**
 * @brief Represent a single neuron in the network
 * @note For similarity purpose, the number of inputs is hardcoded
 */
class Neuron {
    private:
        std::vector<double> weights, gradients;
        std::vector<double> mts; // The first moment of each weight at time t
        std::vector<double> vts; // The second moment of each weight at time t
        double bias_mt, bias_vt; // Moments for this bias
        double bias, z, a;
        double delta; // See: https://towardsdatascience.com/backpropagation-step-by-step-derivation-99ac8fbdcc28/
        double dLda; // Derivative of the loss function with respect to the activation of neuron k in layer l

    public:
        /**
         * @brief Default constructor. Randomize all weights
         * @param n_inputs The number of inputs
         * @param deviation Deviation for the randomized distribution
         */
        Neuron(int n_inputs, double deviation);

        /**
         * @brief Intialize the weights with values
         * @param values To initiate the weights
         */
        Neuron(const std::vector<double>& values);

        /**
         * @param normal_inputs The inputs that feed into this neuron
         * @return The weighted sum for this neuron 
         */
        double weighted_sum(const std::vector<double>& inputs);

        std::vector<double>& get_weights();

        double get_z();

        double get_a();
        void set_a(double value);

        double get_bias();
        void set_bias(double value);

        std::vector<double>& get_gradients();
        std::vector<double>& get_mts();
        std::vector<double>& get_vts();

        double get_delta();
        void set_delta(double value);

        double get_dLda();
        void set_dLda(double value);

        double get_bias_mt();
        void set_bias_mt(double value);

        double get_bias_vt();
        void set_bias_vt(double value);
};