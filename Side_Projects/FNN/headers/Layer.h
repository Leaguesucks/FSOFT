#pragma once

#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <random>

/**
 * @brief CONVENTION: For elemental-wise activation, value > RELU, otherwise < RELU
 */
enum Activation_Type {
    RELU,
    SOFTMAX
};

/**
 * @brief Represent a single neuron layer
 */
class Layer {
    private:
        Activation_Type activation_type; // The type of activaton function this layer use

        // Implement for future optimization
        std::vector<double> weights, gradients; // The weights and gradients of each neuron in this layer
        std::vector<double> mts, vts; // The moments of each neuron in this layer
        std::vector<double> bias_mts, bias_vts; // Moments for the biases of each neuron in this layer
        std::vector<double> biases; // The biases of each neuron in each layer
        std::vector<double> as, zs; // The activation and output of each neuron in this layer
        std::vector<double> deltas, dldas; // The delta and dL / da of each neuron in this layer
        int n_neurons; // The number of neuron in this layer
        int n_inputs; // The number of inputs in each layer

    public:
        /**
         * @brief Default constructor
         * @param n_neurons The number of neuron in this layer
         * @param n_inputs The number of inputs = The number of neuron in the previous layer
         * @param activation_type The type of activation for this layer
         * @param random True to set the weights randomly, false otherwise - to be used by other constructors
         */
        Layer(int n_neurons, int n_inputs, Activation_Type activation_type, bool random);

        /**
         * @brief Constructor for when the weights of the neurons is known
         * @param weights_layer The weights to assign to all neuron in this layer
         * @param activation_type The type fo activation for this layer
         */
        Layer(const std::vector<std::vector<double>>& weights_layer, Activation_Type activation_type);

        /**
         * @brief Constructor for when the weights of the neurons is known
         * @param weights The weights of each neuron in this layer as a flat array
         * @param activation_type The activation type of this layer
         */
        Layer(const std::vector<double>& weights, Activation_Type activation_type);

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
         * @return True if the activation is single element only e.g., RELU, False otherwise e.g., SOFTMAX
         */
        bool is_single_activation();

        /**
         * @brief Calculate the outputs of this layer (activations of all neuron in this layer in other words)
         * @param inputs The inputs to the layer
         */
        void forward(const std::vector<double>& inputs);

        Activation_Type get_activation_type();

        std::vector<double>& get_weights();
        std::vector<double>& get_gradients();
        std::vector<double>& get_mts();
        std::vector<double>& get_vts();

        std::vector<double>& get_bias_mts();
        std::vector<double>& get_bias_vts();
        std::vector<double>& get_biases();
        std::vector<double>& get_as();
        std::vector<double>& get_zs();
        std::vector<double>& get_deltas();
        std::vector<double>& get_dldas();

        double get_n_neurons();
        double get_n_inputs();

    private:
        /**
         * @brief Calculated the weighted sum of each neuron in this layer
         * @param inputs The inputs to the layer
         * @note The sums are returned to zs
         */
        void weighted_sums(const std::vector<double>& inputs);

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

        /** 
         * @param n_inputs The number of inputs
         * @param n_neurons The number of neuron in this layer
         * @return The normal Xavier deviation of this neuron 
         */
        double normal_Xavier_Deviation(int n_inputs, int n_neurons);
};