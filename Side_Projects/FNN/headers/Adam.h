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
        std::size_t t; // Adam time step

    public:
        Adam(Network& network, double a = 0.001, double B1 = 0.9, double B2 = 0.999, double epsilon = 1e-8);

        /**
         * @brief Update the weights in batches
         * @param X The input to feed to the network
         * @param Y The expected output
         */
        void update_weights(const std::vector<double>& X, const std::vector<double>& Y);
};