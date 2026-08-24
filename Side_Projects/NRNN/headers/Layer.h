#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "Neuron.h"

#define INPUT_LAYER 0
#define HIDDEN_LAYER 1
#define OUTPUT_LAYER 2

/**
 * @brief Represent a single neuron layer
 */
class Layer {
    private:
        std::vector<Neuron> neurons;
        int type;

    public:
        /**
         * @brief Default constructor
         * @param n_neurons The number of neuron in this layer
         * @param n_inputs The number of inputs = The number of neuron in the previous layer
         * @param type The type of this layer: INPUT_LAYER | HIDDEN_LAYER | OUTPUT_LAYER
         */
        Layer(int n_neurons, int n_inputs, int type);

        /**
         * @brief Constructor for when the weights of the neurons is known
         * @param weights_layer The weights to assign to all neuron in this layer
         * @param type The type of this layer: INPUT_LAYER | HIDDEN_LAYER | OUTPUT_LAYER
         */
        Layer(const std::vector<std::vector<double>>& weights_layer, int type);

        /** 
         * @param n_inputs The number of inputs
         * @param n_neurons The number of neuron in this layer
         * @return The normal Xavier deviation of this neuron 
         */
        double normal_Xavier_Deviation(int n_inputs, int n_neurons);

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
         * @param inputs The inputs to the layer
         * @return The outputs of this layer as a vector
         */
        std::vector<double> forward(const std::vector<double>& inputs);

        /**
         * @param x The output value of a single neuron
         * @return The derivative of ReLu function
         */
        double d_relu(double x);

        /**
         * @param x The output value of a single neuron
         * @param eX The exponetial sum of all neurons output
         * @param mX The maximum output of all neuron in this network
         * @return The derivative of the softmax function
         */
        double d_softmax(double x, double eX, double mX);

    private:
        /**
         * @param inputs The inputs to the layer
         * @return The weighted sum of this layer
         */
        std::vector<double> weighted_sums(const std::vector<double>& inputs);
};