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
        std::vector<double> weights;
        double bias, z, a, da_dz;

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

        /**
         * @return The weights of this neuron
         */
        std::vector<double>& get_weights();

        double get_z();

        double get_a();
        void set_a(double value);

        double get_da_dz();
        void set_da_dz(double value);
};