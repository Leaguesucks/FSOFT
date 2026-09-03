#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <cstddef>

#include "Network.h"
#include "Layer.h"
#include "Neuron.h"

class Adam {
    private:
        double a; // The learning rate
        double B1, B2; // Decay rates for the moving averages of the gradient
        double epsilon; // To avoid division by zero
        
        Network& network; // The network to train

    public:
        Adam(Network& network, double a = 0.001, double B1 = 0.9, double B2 = 0.999, double epsilon = 1e-8);

        /**
         * @brief Zeros all gradients to prepare them for a new batch
         */
        void zero_gradients();

        /**
         * @brief Call forward and back propagation to calculate the gradients and update the weights
         * @param X The inputs to feed into this network
         * @param Y The expected output of this network
         */
        void accumulate_gradients(const std::vector<double>& X, const std::vector<double>& Y);

        /**
         * @brief Update the weights
         * @param batch_size The size of the batch to use for the update
         */
        void update_weights(size_t batch_size);
};