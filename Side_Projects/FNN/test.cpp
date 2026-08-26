#include "headers/test.h"

void test_1() {
        std::vector<std::vector<double>> hidden_weights = {
        {0.2, -0.4},
        {0.5,  0.3}
    };

    std::vector<std::vector<double>> output_weights = {
        {0.7, -0.2},
        {-0.3, 0.6}
    };

    Layer hidden(hidden_weights, RELU);
    Layer output(output_weights, SOFTMAX);

    Network network(
        std::vector<Layer>{hidden, output},
        SSE
    );

    // Input
    std::vector<double> X = {
        0.8,
        0.5
    };

    // One-hot target
    std::vector<double> Y = {
        1.0,
        0.0
    };

    // ------------------------------------------------------------
    // Forward propagation
    // ------------------------------------------------------------

    std::vector<double> prediction =
        network.forward_propagation(X);

    std::cout << std::fixed << std::setprecision(10);

    std::cout << "Predictions:\n";

    for (double value : prediction)
        std::cout << "  " << value << '\n';

    // Calculate total loss
    double original_loss = 0.0;

    for (size_t i = 0; i < Y.size(); ++i)
        original_loss += network.loss(
            Y[i],
            prediction[i],
            SSE
        );

    std::cout << "\nOriginal loss: "
              << original_loss << "\n\n";

    // ------------------------------------------------------------
    // Test one weight
    //
    // output neuron 0
    // weight 0
    // ------------------------------------------------------------

    const int L = 1;
    const int J = 0;
    const size_t weight_index = 0;

    double analytical_gradient =
        network.d_loss_d_w(L, J, Y)[weight_index];

    // ------------------------------------------------------------
    // Numerical gradient
    //
    // dL/dw ≈ [L(w + epsilon) - L(w - epsilon)] / 2epsilon
    // ------------------------------------------------------------

    const double epsilon = 1e-5;

    std::vector<Neuron>& neurons =
        network.get_layers()[L].get_neurons();

    std::vector<double>& weights =
        neurons[J].get_weights();

    double original_weight = weights[weight_index];

    // L(w + epsilon)
    weights[weight_index] = original_weight + epsilon;

    std::vector<double> prediction_plus =
        network.forward_propagation(X);

    double loss_plus = 0.0;

    for (size_t i = 0; i < Y.size(); ++i)
        loss_plus += network.loss(
            Y[i],
            prediction_plus[i],
            SSE
        );

    // L(w - epsilon)
    weights[weight_index] = original_weight - epsilon;

    std::vector<double> prediction_minus =
        network.forward_propagation(X);

    double loss_minus = 0.0;

    for (size_t i = 0; i < Y.size(); ++i)
        loss_minus += network.loss(
            Y[i],
            prediction_minus[i],
            SSE
        );

    // Restore original weight
    weights[weight_index] = original_weight;

    double numerical_gradient =
        (loss_plus - loss_minus) / (2.0 * epsilon);

    // ------------------------------------------------------------
    // Compare
    // ------------------------------------------------------------

    std::cout << "Gradient test\n";
    std::cout << "-------------\n";

    std::cout << "Layer:              " << L << '\n';
    std::cout << "Neuron:             " << J << '\n';
    std::cout << "Weight:             " << weight_index << '\n';

    std::cout << "Analytical gradient: "
              << analytical_gradient << '\n';

    std::cout << "Numerical gradient:  "
              << numerical_gradient << '\n';

    double absolute_error =
        std::abs(analytical_gradient - numerical_gradient);

    double relative_error =
        absolute_error /
        std::max(
            1.0,
            std::abs(analytical_gradient) +
            std::abs(numerical_gradient)
        );

    std::cout << "Absolute error:      "
              << absolute_error << '\n';

    std::cout << "Relative error:      "
              << relative_error << '\n';

    if (relative_error < 1e-5)
        std::cout << "\nPASS\n";
    else
        std::cout << "\nFAIL\n";
}

double calculate_loss(
    Network& network,
    const std::vector<double>& X,
    const std::vector<double>& Y
) {
    std::vector<double> prediction =
        network.forward_propagation(X);

    double total_loss = 0.0;

    for (size_t i = 0; i < Y.size(); ++i) {
        total_loss += network.loss(
            Y[i],
            prediction[i],
            CATEGORICAL_CROSS_ENTROPY
        );
    }

    return total_loss;
}


// ================================================================
// Gradient check test
//
// Example:
//
// test_gradient_check(
//     64,              // inputs
//     {32, 16, 8},     // hidden layers
//     10               // outputs
// );
//
// Architecture:
//
// 64
//  ↓
// 32 ReLU
//  ↓
// 16 ReLU
//  ↓
// 8 ReLU
//  ↓
// 10 Softmax
// ================================================================

void test_gradient_check(
    size_t input_size,
    const std::vector<size_t>& hidden_sizes,
    size_t output_size
) {
    // ============================================================
    // Configuration
    // ============================================================

    constexpr double epsilon = 1e-5;
    constexpr double tolerance = 1e-5;

    std::cout << std::fixed << std::setprecision(10);

    std::cout << "========================================\n";
    std::cout << "Gradient Check\n";
    std::cout << "========================================\n\n";

    std::cout << "Input neurons:  "
              << input_size << '\n';

    std::cout << "Hidden layers:  "
              << hidden_sizes.size() << '\n';

    for (size_t i = 0; i < hidden_sizes.size(); ++i) {
        std::cout
            << "  Layer " << i
            << ": "
            << hidden_sizes[i]
            << " ReLU neurons\n";
    }

    std::cout << "Output neurons: "
              << output_size
              << " Softmax neurons\n\n";


    // ============================================================
    // Validate architecture
    // ============================================================

    if (input_size == 0) {
        throw std::invalid_argument(
            "Input size must be greater than zero"
        );
    }

    if (output_size == 0) {
        throw std::invalid_argument(
            "Output size must be greater than zero"
        );
    }

    for (size_t size : hidden_sizes) {
        if (size == 0) {
            throw std::invalid_argument(
                "Hidden layer size must be greater than zero"
            );
        }
    }


    // ============================================================
    // Create layers
    // ============================================================

    std::vector<Layer> layers;

    size_t previous_size = input_size;

    for (size_t layer_index = 0;
         layer_index < hidden_sizes.size();
         ++layer_index) {

        size_t current_size =
            hidden_sizes[layer_index];

        std::vector<std::vector<double>> weights(
            current_size,
            std::vector<double>(previous_size)
        );

        // --------------------------------------------------------
        // Deterministic weights
        //
        // Small values prevent extreme activations.
        // --------------------------------------------------------

        for (size_t neuron = 0;
             neuron < current_size;
             ++neuron) {

            for (size_t weight = 0;
                 weight < previous_size;
                 ++weight) {

                double value =
                    static_cast<double>(
                        (
                            (layer_index + 1) * 17
                            + (neuron + 1) * 31
                            + (weight + 1) * 13
                        ) % 200
                    ) / 1000.0;

                value -= 0.1;

                weights[neuron][weight] =
                    value;
            }
        }

        layers.emplace_back(
            weights,
            RELU
        );

        previous_size = current_size;
    }


    // ============================================================
    // Output layer
    // ============================================================

    {
        std::vector<std::vector<double>> weights(
            output_size,
            std::vector<double>(previous_size)
        );

        for (size_t neuron = 0;
             neuron < output_size;
             ++neuron) {

            for (size_t weight = 0;
                 weight < previous_size;
                 ++weight) {

                double value =
                    static_cast<double>(
                        (
                            101
                            + (neuron + 1) * 23
                            + (weight + 1) * 37
                        ) % 160
                    ) / 1000.0;

                value -= 0.08;

                weights[neuron][weight] =
                    value;
            }
        }

        layers.emplace_back(
            weights,
            SOFTMAX
        );
    }


    // ============================================================
    // Create network
    // ============================================================

    Network network(
        layers,
        CATEGORICAL_CROSS_ENTROPY
    );


    // ============================================================
    // Deterministic input
    // ============================================================

    std::vector<double> X(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        X[i] = std::sin(static_cast<double>(i) * 0.37);
    }


    // ============================================================
    // One-hot target
    //
    // Pick a deterministic class.
    // ============================================================

    std::vector<double> Y(
        output_size,
        0.0
    );

    size_t target_class =
        output_size / 2;

    Y[target_class] = 1.0;


    // ============================================================
    // Forward propagation
    // ============================================================

    std::vector<double> prediction =
        network.forward_propagation(X);

    std::cout << "Network output:\n";

    for (size_t i = 0;
         i < prediction.size();
         ++i) {

        std::cout
            << "  ["
            << i
            << "] "
            << prediction[i]
            << '\n';
    }


    // ============================================================
    // Verify Softmax
    // ============================================================

    double probability_sum = 0.0;

    for (double p : prediction)
        probability_sum += p;

    std::cout << "\nProbability sum: "
              << probability_sum
              << '\n';


    // ============================================================
    // Original loss
    // ============================================================

    double original_loss = calculate_loss(network, X, Y);

    std::cout << "Original loss:   "
              << original_loss
              << "\n\n";


    // ============================================================
    // Gradient checking
    // ============================================================

    size_t total_tests = 0;
    size_t failed_tests = 0;

    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;

    size_t worst_L = 0;
    size_t worst_J = 0;
    size_t worst_I = 0;

    double worst_analytical = 0.0;
    double worst_numerical = 0.0;


    // ============================================================
    // Iterate over every layer
    // ============================================================

    for (size_t L = 0; L < network.get_layers().size(); ++L) {
        std::vector<Neuron>& neurons = network.get_layers()[L].get_neurons();

        // ========================================================
        // Every neuron
        // ========================================================

        for (size_t J = 0; J < neurons.size(); ++J) {
            std::vector<double>& weights = neurons[J].get_weights();

            // ----------------------------------------------------
            // Calculate analytical gradients
            // ----------------------------------------------------

            std::vector<double> analytical_gradients =
                network.d_loss_d_w(static_cast<int>(L), static_cast<int>(J), Y);

            // ====================================================
            // Every weight
            // ====================================================

            for (size_t I = 0; I < weights.size(); ++I) {
                ++total_tests;

                double original_weight =
                    weights[I];


                // =================================================
                // L(w + epsilon)
                // =================================================

                weights[I] =
                    original_weight + epsilon;

                double loss_plus = calculate_loss(network, X, Y);

                // =================================================
                // L(w - epsilon)
                // =================================================

                weights[I] =
                    original_weight - epsilon;

                double loss_minus = calculate_loss(network, X, Y);


                // =================================================
                // Restore weight
                // =================================================

                weights[I] = original_weight;


                // =================================================
                // Numerical gradient
                // =================================================

                double numerical_gradient = (loss_plus - loss_minus) / (2.0 * epsilon);

                // =================================================
                // Analytical gradient
                // =================================================

                double analytical_gradient = analytical_gradients[I];


                // =================================================
                // Error
                // =================================================

                double absolute_error = std::abs(analytical_gradient - numerical_gradient);
                double denominator = std::abs(analytical_gradient) + std::abs(numerical_gradient);
                double relative_error;

                if (denominator < 1e-12) {
                    relative_error = absolute_error;
                } else {
                    relative_error = absolute_error / denominator;
                }


                // =================================================
                // Track worst gradient
                // =================================================

                if (relative_error > max_relative_error) {
                    max_relative_error = relative_error;

                    max_absolute_error = absolute_error;

                    worst_L = L;
                    worst_J = J;
                    worst_I = I;

                    worst_analytical = analytical_gradient;

                    worst_numerical = numerical_gradient;
                }

                // =================================================
                // Check tolerance
                // =================================================

                if (relative_error > tolerance) {
                    ++failed_tests;
                }
            }
        }
    }


    // ============================================================
    // Results
    // ============================================================

    std::cout << "Gradient check\n";
    std::cout << "---------------\n";

    std::cout << "Total weights tested: "
              << total_tests
              << '\n';

    std::cout << "Failed tests:         "
              << failed_tests
              << '\n';

    std::cout << "\nWorst gradient:\n";

    std::cout << "  Layer:              "
              << worst_L
              << '\n';

    std::cout << "  Neuron:             "
              << worst_J
              << '\n';

    std::cout << "  Weight:             "
              << worst_I
              << '\n';

    std::cout << "  Analytical:         "
              << worst_analytical
              << '\n';

    std::cout << "  Numerical:          "
              << worst_numerical
              << '\n';

    std::cout << "  Absolute error:     "
              << max_absolute_error
              << '\n';

    std::cout << "  Relative error:     "
              << max_relative_error
              << '\n';


    // ============================================================
    // Final result
    // ============================================================

    if (failed_tests == 0) {
        std::cout
            << "\nPASS: "
            << "All gradients match.\n";

    } else {
        std::cout
            << "\nFAIL: "
            << "Some gradients do not match.\n";
    }
}