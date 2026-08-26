#pragma once

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

double calculate_loss(
    Network& network,
    const std::vector<double>& X,
    const std::vector<double>& Y
);

void test_1();

void test_gradient_check(
    size_t input_size,
    const std::vector<size_t>& hidden_sizes,
    size_t output_size
);