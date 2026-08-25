#pragma once

#include <vector>
#include <stdexcept>

#include "Layer.h"

/**
 * @brief A neuron network with an input layer, hidden layers and output layers
 */
class Network {
    private:
        std::vector<Layer> layers;
        std::vector<double> inputs;

    public:
        /**
         * @brief The default constructor
         * @param architecture Each number in this vector represent the number of neuron in each layer.
         *                 The first and last member is the input and output layer respectively
         */
        Network(const std::vector<int>& architecture);

        /**
         * @brief Use this constructor if there has been a mid-training
         * @param layers The layer with all the weights initialized
         */
        Network(const std::vector<Layer>& layers);

        /**
         * @brief Feed the inputs through the network and return the results
         * @param inputs The inputs to feed to the network
         * @return The probabilities of each output
         */
        std::vector<double> forward_propagation(const std::vector<double>& inputs);

        /**
         * @brief The loss function used for this network
         * @param y The desired output that should be produced by a single neuron
         * @param y_hat The actual output produced by a single neuron
         * @return The loss value (cost)
         */
        double loss(double y, double y_hat);

        /**
         * @param y The desired output
         * @param y_hat The actual output
         * @return The derivative of the loss function
         */
        double d_loss(double y, double y_hat);

        /**
         * @param L The <L>-th layer
         * @param J The <J>-th neuron
         * @param Y The expected outputs
         * @return The gradient of this neuron's weights to the loss function
         */
        std::vector<double> d_loss_d_w(int L, int J, const std::vector<double>& Y);

        /**
         * @brief Recursively calculate the gradient of the activation of the <J> neuron in the <L> layer
         * @param L The <L>-th layer
         * @param J The <J>-th neuron
         * @param Y The expected outputs of the network
         * @return The gradient of the activation of this neuron
         */
        double d_loss_d_a(int L, int J, const std::vector<double>& Y);
};