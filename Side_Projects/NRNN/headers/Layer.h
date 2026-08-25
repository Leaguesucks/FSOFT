#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "Neuron.h"

enum Layer_Type {
    HIDDEN_LAYER,
    OUTPUT_LAYER
};

enum Activation_Type {
    RELU,
    SOFTMAX
};

/**
 * @brief Represent a single neuron layer
 */
class Layer {
    private:
        std::vector<Neuron> neurons;
        std::vector<double> additional_data; // Holds addtional data for back propagation, such as max_a and sum (exp(a))
        Layer_Type type;

    public:
        /**
         * @brief Default constructor
         * @param n_neurons The number of neuron in this layer
         * @param n_inputs The number of inputs = The number of neuron in the previous layer
         * @param type The type of this layer: HIDDEN_LAYER | OUTPUT_LAYER
         */
        Layer(int n_neurons, int n_inputs, Layer_Type type);

        /**
         * @brief Constructor for when the weights of the neurons is known
         * @param weights_layer The weights to assign to all neuron in this layer
         * @param type The type of this layer: HIDDEN_LAYER | OUTPUT_LAYER
         */
        Layer(const std::vector<std::vector<double>>& weights_layer, Layer_Type type);

        /** 
         * @param n_inputs The number of inputs
         * @param n_neurons The number of neuron in this layer
         * @return The normal Xavier deviation of this neuron 
         */
        double normal_Xavier_Deviation(int n_inputs, int n_neurons);

        /**
         * @param args The argunments to pass to the activation function
         * @param activate_type The activation function to use: RELU | SOFTMAX
         * @return The activation of a single neuron is this layer
         */
        double activate(const std::vector<double>& arg, Activation_Type activate_type);

        /**
         * @param x The input to the activation function
         * @param activate_type The activation function to use
         * @return The activation of a single neuron is this layer
         * @note Since the activation functions in this overload doesn't rely on other z inputs,
         *       we only need to return a single case where i = j
         */
        double activate(double x, Activation_Type activation_type);

        /**
         * @param i The i-th index for neuron's activation
         * @param j The j-th index for neuron's output
         * @param additional_args Additional argument
         * @param activation_type The type of activation function
         * @return The derivative of the activation of the i-th neuron with respect to the 
         */
        double d_activate(int i, int j, const std::vector<double>& additional_args, Activation_Type activation_type);

        /**
         * @param inputs The inputs to the layer
         * @return The outputs of this layer as a vector
         */
        std::vector<double> forward(const std::vector<double>& inputs);

        std::vector<Neuron>& get_neurons();

        Layer_Type get_type();

        std::vector<double>& get_additional_data();

    private:
        /**
         * @param inputs The inputs to the layer
         * @return The weighted sum of this layer
         */
        std::vector<double> weighted_sums(const std::vector<double>& inputs);

        /**
         * @brief Activation function
         * @param x The input value
         * @return 0 or x, whichever larger
         * @note Usually use this for the hidden layer
         */
        double relu(double x);

        /**
         * @brief Activation function
         * @param x The input value of this neuron
         * @param eX The sum of exponent output of all neuron in this layer
         * @param mX The maximum output in this layer
         * @return The activation probabilties of this neuron
         * @note Usually use this for the output layer
         */
        double softmax(double x, double eX, double mX);

        /**
         * @param x The output value of a single neuron
         * @return The derivative of ReLu function
         */
        double d_relu(double x);

        /**
         * @param i The i-th index for neuron's activation
         * @param j The j-th index for neuron's output
         * @return The derivative of the softmax function
         */
        double d_softmax(int i, int j);
};