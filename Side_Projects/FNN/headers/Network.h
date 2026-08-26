#pragma once

#include <vector>
#include <stdexcept>

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
        std::vector<double> inputs;
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
         * @return The probabilities of each output
         */
        std::vector<double> forward_propagation(const std::vector<double>& inputs);

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

        /**
         * @param X The inputs to feed into this network
         * @param Y The expectedd output
         * @return The total loss
         */
        double total_loss(const std::vector<double>& X, const std::vector<double>& Y);

        /**
         * @brief Update the weights of a neuron J in layer L using back propagation
         * @param L The <L>-th layer
         * @param j The <J>-th layer
         * @param learning_rate The learning rate to apply: w_new = w_old - lr * grad
         * @param Y The expected output
         */
        void update_weights(int L, int J, double learning_rate, const std::vector<double>& Y);

        std::vector<Layer>& get_layers();

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