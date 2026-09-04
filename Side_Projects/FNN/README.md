# Neural Network From Scratch in C++
**JUST AS GOD INTENDED**

A feed-forward neural network implemented **from scratch in C++**, trained on the **MNIST handwritten digit dataset**.

The primary goal of this project is not simply to achieve high classification accuracy, but to understand how neural networks work internally by implementing the major components without relying on machine-learning frameworks.

The project currently supports forward propagation, backpropagation, categorical cross-entropy, Softmax, ReLU, mini-batch training, Adam optimization, gradient checking, model serialization, and MNIST inference.

---

## Features

* Fully-connected feed-forward neural network
* Forward propagation
* Backpropagation
* ReLU activation
* Softmax activation
* Categorical Cross-Entropy loss
* Sum of Squared Errors (SSE)
* Binary Cross-Entropy (BCE)
* Adam optimizer
* Mini-batch gradient descent
* Gradient checking using numerical differentiation
* Xavier weight initialization
* Bias parameters
* Adam first and second moments
* Model serialization and loading
* MNIST dataset loader
* Randomized training batches
* CPU-based inference
* Python tools for visualizing inference results

The implementation intentionally avoids high-level machine-learning frameworks so that the underlying mathematics and algorithms remain visible in the code.

---

## Network Architecture

The current network used for MNIST classification is:

```text
Input
784 neurons
   │
   ▼
Fully Connected
128 neurons
ReLU
   │
   ▼
Fully Connected
64 neurons
ReLU
   │
   ▼
Fully Connected
10 neurons
Softmax
   │
   ▼
Output
10 class probabilities
```

The 784 inputs correspond to the `28 × 28` pixels of an MNIST image.

The output layer contains 10 neurons, representing digits:

```text
0 1 2 3 4 5 6 7 8 9
```

The Softmax layer converts the final logits into a probability distribution.

---

## Mathematical Implementation

### Fully Connected Layer

For a neuron:

$$
z_i = \sum_j w_{ij}x_j + b_i
$$

The activation is then applied:

$$
a_i = f(z_i)
$$

---

### ReLU

Hidden layers currently use ReLU:

$$
\operatorname{ReLU}(x) = \max(0,x)
$$

Its derivative is:

$$
\operatorname{ReLU}'(x) =
\begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$

---

### Softmax

The output layer uses numerically stable Softmax:

$$
\operatorname{softmax}(z_i)
=
\frac{e^{z_i-z_{\max}}}
{\sum_j e^{z_j-z_{\max}}}
$$

where

$$
z_{\max} = \max_j z_j
$$

Subtracting the maximum logit does not change the resulting probabilities, but prevents unnecessary numerical overflow during exponentiation.

---

### Categorical Cross-Entropy

For a one-hot target vector:

$$
L = -\sum_i y_i \log(\hat{y}_i)
$$

Because the target vector is one-hot, this reduces to:

$$
L = -\log(\hat{y}_{\mathrm{correct}})
$$

The derivative with respect to the Softmax output is:

$$
\frac{\partial L}{\partial \hat{y}_i}
=
-\frac{y_i}{\hat{y}_i}
$$

The current implementation supports the general Softmax Jacobian during backpropagation.

For Softmax combined with categorical cross-entropy, the derivative can be simplified to the well-known expression:

$$
\frac{\partial L}{\partial z_i}
=
\hat{y}_i-y_i
$$

This avoids explicitly constructing or multiplying by the Softmax Jacobian and therefore substantially reduces the computational cost of the output-layer derivative.

---

## Backpropagation

The network computes gradients using the chain rule.

For each layer, the gradient with respect to a weight is:

$$
\frac{\partial L}{\partial w_{ij}}
=
\delta_i a_j
$$

where:

* $\delta_i$ is the error signal for neuron $i$
* $a_j$ is the activation from the previous layer

The bias gradient is:

$$
\frac{\partial L}{\partial b_i}
=
\delta_i
$$

Gradients are accumulated across a mini-batch before the optimizer updates the parameters.

---

## Adam Optimizer

Training uses the Adam optimization algorithm.

For each parameter, the first and second moments are updated as:

$$
m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1)g_t
$$

$$
v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2)g_t^2
$$

Bias correction is then applied:

$$
\hat{m}_t
=
\frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t
=
\frac{v_t}{1-\beta_2^t}
$$

The parameter is then updated using:

$$
\theta_t
=
\theta_{t-1}
-
\alpha
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\epsilon}
$$

Default parameters:

```text
Learning rate:  0.001
β1:             0.9
β2:             0.999
ε:              1e-8
```

---

## Mini-Batch Training

The network uses mini-batch gradient descent.

For each batch:

```text
1. Clear accumulated gradients
2. Forward propagate every sample
3. Backpropagate every sample
4. Accumulate gradients
5. Average gradients over the batch
6. Perform one Adam update
```

The Adam timestep is incremented **once per mini-batch**, rather than once per individual training sample.

---

## Gradient Checking

One of the most important parts of the project is numerical gradient checking.

The analytical gradient produced by backpropagation is compared against a numerical approximation using the central difference method:

$$
\frac{\partial L}{\partial w}
\approx
\frac{
L(w+\epsilon)-L(w-\epsilon)
}{
2\epsilon
}
$$

This was used to verify both the backpropagation implementation and the flattened weight representation.

A full gradient check produced:

```text
Total weights tested: 242304
Failed tests:         0

Worst gradient:
  Layer:              3
  Neuron:             5
  Weight:             55
  Analytical:        -0.0838815663
  Numerical:         -0.0838815671
  Absolute error:     0.0000000009
  Relative error:     0.0000000103

PASS: All weight gradients match.
```

This provides strong evidence that the implemented weight gradients are mathematically correct.\

---

## Memory Layout

The network stores layer parameters using contiguous `std::vector<double>` arrays rather than individual `Neuron` objects.

For a layer containing `N` neurons and `M` inputs, weights are stored as:

```text
[w00, w01, w02, ..., w0M,
 w10, w11, w12, ..., w1M,
 ...
 wN0, wN1, wN2, ..., wNM]
```

The weight for neuron `i` and input `j` is accessed using:

```cpp
weights[i * n_inputs + j]
```

This representation was chosen to improve:

* Memory locality
* Cache utilization
* Allocation behavior
* Ease of vectorization
* Future SIMD optimization
* Overall performance compared with an object-per-neuron design

The project originally used a `Neuron` class but was later redesigned around contiguous layer-level arrays.

---

## MNIST Results

The current `784 → 128 → 64 → 10` network achieves approximately:

```text
Test accuracy: 98.12%
```

on the MNIST test set.

This is a, in my humble opinion, decent result for a relatively simple fully-connected network implemented entirely from scratch.

A fully-connected network does not explicitly exploit the spatial structure of images. Each pixel is treated as an independent input feature, unlike a convolutional neural network.

For this reason, increasing the size of the fully-connected network does not necessarily produce proportional improvements in accuracy.

---

## Why MNIST?

MNIST provides a useful environment for implementing a neural network from first principles because it is:

* Small enough to train on a CPU
* Easy to visualize
* Well understood
* Large enough to expose implementation bugs
* Simple enough to make numerical debugging practical

---

## Project Structure

A typical project layout is:

```text
.
├── headers/
│   ├── Adam.h
│   ├── Layer.h
│   ├── Loss.h
│   ├── MNIST.h
│   └── Network.h
│
├── src/
│   ├── Adam.cpp
│   ├── Layer.cpp
│   ├── Loss.cpp
│   ├── MNIST.cpp
│   └── Network.cpp
│
├── tests/
│   └── ...
│
├── mnist/
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   └── t10k-labels.idx1-ubyte
│
├── training/
│   └── mnist_train.bin
│
├── python/
│   └── ...
│
├── main.cpp
└── README.md
```

The exact directory structure may differ depending on the current build configuration.

---

## Building

The project is written in C++ and requires a compiler with **C++17** support.

First, make the setup script executable:

```bash
chmod +x setup.sh
```

Then run it to install the required dependencies:

```bash
./setup.sh
```

Once the dependencies have been installed, compile the project using:

```bash
make
```

The compiled program can then be executed with:

```bash
./bin/main.exe
```

### MNIST Display

If you want a more interactive way to inspect the network's predictions, you can use `MNIST_Display.py`.

First, activate the Python virtual environment:

```bash
source .venv/bin/activate
```

You can then choose between two modes.

**Manual mode**

```bash
python3 MNIST_Display.py MANUAL
```

This allows you to inspect the MNIST images one at a time and examine the network's predictions.

> **Note:** For some damn reason, my WSL installation refuses to properly display GUI applications. Instead of opening a window, the program generates the results as PNG images in the `mnist_results/` directory.

**Summary mode**

```bash
python3 MNIST_Display.py SUMMARY
```

This produces a summary of the network's performance, including its accuracy and prediction results.

---

## Dataset

The project uses the MNIST dataset.

The training set contains:

```text
60,000 training images
10,000 test images
28 × 28 pixels
10 classes
```

Each image is represented as 784 input values.

Pixel values are normalized to:

```text
[0, 255] → [0.0, 1.0]
```

The labels are converted to one-hot vectors.

For example, the label `7` becomes:

```text
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```

---

## Model Serialization

The network can save its parameters to a binary file and load them later.

The serialized model contains:

```text
Network metadata
├── Version
├── Input size
├── Number of layers
├── Loss type
└── Adam timestep

For each layer
├── Number of neurons
├── Activation type
├── Number of inputs
├── Weights
├── Biases
├── Adam first moments
├── Adam second moments
├── Bias first moments
└── Bias second moments
```

Saving the Adam state allows training to be resumed without losing the optimizer's accumulated momentum information.

---

## Inference

After training, the network produces a probability for each digit.

For example:

```text
0: 0.00000000
1: 0.00000000
2: 0.00000000
3: 0.00000002
4: 0.00000000
5: 0.00000000
6: 0.00000000
7: 0.99999998
8: 0.00000000
9: 0.00000000
```

The predicted class is the index with the highest probability.

```cpp
size_t predicted_label = std::distance(
    Y_HAT.begin(),
    std::max_element(Y_HAT.begin(), Y_HAT.end())
);
```

---

## Current Limitations

This project is intentionally simple and is not intended to compete with production machine-learning libraries.

Current limitations include:

* CPU-only computation
* Fully-connected layers only
* No convolutional layers
* No GPU acceleration
* No SIMD implementation yet
* No multithreaded training
* No dropout
* No batch normalization
* No learning-rate scheduler
* No automatic early stopping
* Limited data augmentation

These limitations also provide opportunities for future optimization and experimentation.

---

## Future Work

The project is primarily an educational and experimental implementation, so there are several planned directions.

### Performance

* SIMD/vectorized matrix operations
* Better cache utilization
* Multithreaded training
* Reduce temporary allocations
* Improve memory layout
* Parallel batch processing
* Benchmark different layer implementations

### Training

* Learning-rate scheduling
* L2 weight decay
* Early stopping
* Better initialization strategies
* Data augmentation
* Hard-example mining
* More detailed validation metrics

### Neural Network Features

* Convolutional layers
* Max pooling
* Leaky ReLU
* Tanh
* Sigmoid
* Additional loss functions
* Dropout
* Batch normalization

### Architecture

The eventual goal is to experiment with a much more performance-oriented architecture suitable for neural-network-based chess evaluation.

In particular, this project provides a foundation for experimenting with **NNUE-style networks**, where memory layout, incremental evaluation, and CPU efficiency become especially important.

---

## Lessons Learned

This project has been primarily about understanding what happens inside a neural network rather than treating a neural network as a black box.

Some of the most important lessons so far include:

1. **Forward propagation is relatively straightforward.**
2. **Backpropagation is where indexing and mathematical mistakes become difficult to debug.**
3. **Numerical gradient checking is extremely valuable.**
4. **A mathematically correct implementation should be verified before optimizing it.**
5. **Memory layout can matter as much as the mathematical algorithm when implementing neural networks in C++.**
6. **A larger network does not automatically produce better results.**
7. **The activation function and loss function should be considered together.**
8. **Mini-batch Adam requires careful handling of gradient accumulation and optimizer timesteps.**
9. **Numerical stability matters, particularly for Softmax and Cross-Entropy.**
10. **Once correctness is established, the implementation can be optimized without changing its mathematical behavior.**

---

## Goals of the Project

The long-term purpose of this project is to progressively move from a simple educational neural network toward a high-performance neural-network implementation.

The progression is roughly:

```text
Mathematical understanding
        ↓
Basic neural network
        ↓
Backpropagation
        ↓
Gradient verification
        ↓
Adam optimization
        ↓
Mini-batch training
        ↓
Efficient memory layout
        ↓
SIMD / parallelization
        ↓
High-performance inference
```

The emphasis throughout the project is:

> **Understand it first. Optimize it second.**
> **WHO THE F NEEDS TORCH WHEN WE CAN WRITE OUR OWN NEURON NETWORK**

---

## License

This project is intended primarily for educational and experimental purposes.

You can do whaterver the f you want with it.