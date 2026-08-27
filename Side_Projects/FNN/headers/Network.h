#pragma once

#include <vector>
#include <stdexcept>
#include <string>
#include <fstream>
#include <cstdint>

#include "Layer.h"

enum Loss_Type {
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY,
    SSE
};

struct Layer_Architecture {
    int n_neurons; // The number of neuron in this layer == The number of outputs if this is the output layer
    Activation_Type activation_type;
};

/**
 * @brief A neuron network with an input layer, hidden layers and output layers
 */
class Network {
    private:
        std::vector<Layer> layers;
        std::vector<double> X; // The inputs to feed into this network
        std::vector<double> Y_HAT; // The outputs of this network
        Loss_Type loss_type;

    public:
        /**
         * @brief The default constructor
         * @param n_inputs The number of inputs to feed into this network
         * @param architectures The architecture of each layer in this network
         * @param loss_type The type of loss function this network use
         */
        Network(int n_inputs, const std::vector<Layer_Architecture>& architectures, Loss_Type loss_type);

        /**
         * @brief Use this constructor if there has been a mid-training
         * @param layers The layer with all the weights initialized
         * @param loss_type The type of loss function this network use
         */
        Network(const std::vector<Layer>& layers, Loss_Type loss_type);

        /**
         * @brief Feed the inputs through the network and return the results
         * @param inputs The inputs to feed to the network
         */
        void forward_propagation(const std::vector<double>& inputs);

        /**
         * @brief The loss function used for this network
         * @param y The desired output that should be produced by a single neuron
         * @param y_hat The actual output produced by a single neuron
         * @param type The type of loss function
         * @return The loss value (cost)
         */
        double loss(double y, double y_hat, Loss_Type type);

        /**
         * @param y The desired output
         * @param y_hat The actual output
         * @param type The type of loss function
         * @return The derivative of the loss function
         */
        double d_loss(double y, double y_hat, Loss_Type type);

        /**
         * @param X The inputs to feed into this network
         * @param Y The expected output
         * @return The total loss
         */
        double total_loss(const std::vector<double>& X, const std::vector<double>& Y);

        std::vector<Layer>& get_layers();

        /**
         * @brief Calculate the gradient of the weights, dL/da and the delta terms for each neuron
         * @param Y The expected outputs
         * @note Use this only after foward propagation has been called
         * @note This back propagation is meant for general case. Should look into sepcial cases or
         *       parallelism for optimization later
         */
        void back_propagation(const std::vector<double>& Y);

        /**
         * @brief Perform both forward and backward propagation in one cycle
         * @param X The inputs to feed into this network
         * @param Y The expected output of this network
         */
        void forward_back_propagation(const std::vector<double>& X, const std::vector<double>& Y);

        std::vector<double>& get_Y_HAT();

        /**
         * @brief Save the current network to a file for next use
         * @param filename Path to the file to save
         */
        void save(const std::string& filename);

        /**
         * @brief Load an already existed network
         * @param filename Path to the saved file
         */
        void load(const std::string& filename);

    private:
        /**
         * @param y The expected output
         * @param y_hat The actual output
         * @return The binary cross entropy loss
         */
        double binary_cross_entropy(double y, double y_hat);

        /**
         * @param y The expected output
         * @param y_hat The actual output
         * @return The categorical cross entropy loss
         */
        double categorical_cross_entropy(double y, double y_hat);

        /**
         * @param y The expected output
         * @param y_hat The actual output
         * @return The sum squared error
         */
        double sse(double y, double y_hat);

        /**
         * @param y The expected output
         * @param y_hat The actual output
         * @return The derivative of the binary cross entropy loss function
         */
        double d_binary_cross_entropy(double y, double y_hat);

        /**
         * @param y The expectedd output
         * @param y_hat The actual output
         * @return The derivative of the categorical entropy loss
         */
        double d_categorical_cross_entropy(double y, double y_hat);

        /**
         * @param y The expected output
         * @param y_hat The actual output
         * @return The derivative of the sum square error function
         */
        double d_sse(double y, double y_hat);
};