import sys
import os
import struct
import math

import numpy as np
from PIL import Image


# ============================================================
# Configuration
# ============================================================

NETWORK_FILE = "training/mnist_train.bin"

MNIST_IMAGES = "mnist/t10k-images.idx3-ubyte"
MNIST_LABELS = "mnist/t10k-labels.idx1-ubyte"

OUTPUT_DIR = ".out"

WORST_LOSS_FILE = "training/worst_loss.png"
WORST_CONFIDENCE_FILE = "training/worst_confidence.png"

INPUT_SIZE = 784
OUTPUT_SIZE = 10

NUM_CLASSES = 10


# ============================================================
# Directories
# ============================================================

def ensure_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("training", exist_ok=True)


# ============================================================
# MNIST loader
# ============================================================

def load_mnist_images(filename):
    print(f"Loading images: {filename}")

    with open(filename, "rb") as f:
        header = f.read(16)

        if len(header) != 16:
            raise RuntimeError(
                "Could not read MNIST image header"
            )

        magic, count, rows, cols = struct.unpack(
            ">IIII",
            header
        )

        if magic != 2051:
            raise ValueError(
                f"Invalid MNIST image magic number: {magic}"
            )

        raw = f.read()

    expected_size = count * rows * cols

    if len(raw) != expected_size:
        raise ValueError(
            f"Invalid MNIST image file size. "
            f"Expected {expected_size} bytes, "
            f"got {len(raw)}"
        )

    images = np.frombuffer(
        raw,
        dtype=np.uint8
    ).reshape(
        count,
        rows,
        cols
    )

    return images


def load_mnist_labels(filename):
    print(f"Loading labels: {filename}")

    with open(filename, "rb") as f:
        header = f.read(8)

        if len(header) != 8:
            raise RuntimeError(
                "Could not read MNIST label header"
            )

        magic, count = struct.unpack(
            ">II",
            header
        )

        if magic != 2049:
            raise ValueError(
                f"Invalid MNIST label magic number: {magic}"
            )

        raw = f.read()

    if len(raw) != count:
        raise ValueError(
            f"Invalid MNIST label file size. "
            f"Expected {count} bytes, "
            f"got {len(raw)}"
        )

    labels = np.frombuffer(
        raw,
        dtype=np.uint8
    )

    return labels


def load_mnist():
    images = load_mnist_images(
        MNIST_IMAGES
    )

    labels = load_mnist_labels(
        MNIST_LABELS
    )

    if len(images) != len(labels):
        raise ValueError(
            "Number of images and labels do not match"
        )

    print(
        f"Loaded {len(images)} test images."
    )

    return images, labels


# ============================================================
# Binary reader
# ============================================================

def read_u32(f):
    data = f.read(4)

    if len(data) != 4:
        raise EOFError(
            "Unexpected EOF while reading uint32"
        )

    return struct.unpack(
        "<I",
        data
    )[0]


def read_u64(f):
    data = f.read(8)

    if len(data) != 8:
        raise EOFError(
            "Unexpected EOF while reading uint64"
        )

    return struct.unpack(
        "<Q",
        data
    )[0]


def read_double(f):
    data = f.read(8)

    if len(data) != 8:
        raise EOFError(
            "Unexpected EOF while reading double"
        )

    return struct.unpack(
        "<d",
        data
    )[0]


# ============================================================
# Network classes
# ============================================================

class Neuron:

    def __init__(self, weights, bias):

        self.weights = np.asarray(
            weights,
            dtype=np.float64
        )

        self.bias = float(bias)


class Layer:

    def __init__(self, neurons, activation):

        self.neurons = neurons
        self.activation = activation


class Network:

    RELU = 0
    SOFTMAX = 1

    def __init__(self, input_size, layers):

        self.input_size = input_size
        self.layers = layers

    def forward(self, x):

        x = np.asarray(
            x,
            dtype=np.float64
        )

        if x.size != self.input_size:
            raise ValueError(
                f"Network expects {self.input_size} "
                f"inputs, got {x.size}"
            )

        a = x

        # ----------------------------------------------------
        # Forward through every layer
        # ----------------------------------------------------

        for layer_index, layer in enumerate(
            self.layers
        ):

            z = np.empty(
                len(layer.neurons),
                dtype=np.float64
            )

            for neuron_index, neuron in enumerate(
                layer.neurons
            ):

                z[neuron_index] = (
                    np.dot(
                        neuron.weights,
                        a
                    )
                    + neuron.bias
                )

            # ------------------------------------------------
            # ReLU
            # ------------------------------------------------

            if layer.activation == self.RELU:

                a = np.maximum(
                    0.0,
                    z
                )

            # ------------------------------------------------
            # Softmax
            # ------------------------------------------------

            elif layer.activation == self.SOFTMAX:

                # Same numerical stabilization normally used
                # by a C++ implementation:
                #
                # exp(z_i - max(z))
                #
                # This prevents overflow.

                z_max = np.max(z)

                exp_z = np.exp(
                    z - z_max
                )

                denominator = np.sum(
                    exp_z
                )

                if denominator == 0.0:
                    raise FloatingPointError(
                        "Softmax denominator is zero"
                    )

                a = exp_z / denominator

            else:

                raise ValueError(
                    f"Unsupported activation "
                    f"{layer.activation} "
                    f"in layer {layer_index}"
                )

        return a


# ============================================================
# Network loader
# ============================================================

def load_network(filename):

    print(
        f"Loading network: {filename}"
    )

    with open(filename, "rb") as f:

        # ----------------------------------------------------
        # Magic
        # ----------------------------------------------------

        magic = f.read(3)

        if magic != b"FNN":
            raise ValueError(
                f"Invalid network magic: {magic}"
            )

        # ----------------------------------------------------
        # Version
        # ----------------------------------------------------

        version = read_u32(f)

        if version != 1:
            raise ValueError(
                f"Unsupported network version: {version}"
            )

        # ----------------------------------------------------
        # Input size
        # ----------------------------------------------------

        input_size = read_u64(f)

        # ----------------------------------------------------
        # Number of layers
        # ----------------------------------------------------

        n_layers = read_u64(f)

        print(
            f"Input size: {input_size}"
        )

        print(
            f"Number of layers: {n_layers}"
        )

        layers = []

        # ----------------------------------------------------
        # Read every layer
        # ----------------------------------------------------

        for layer_index in range(n_layers):

            n_neurons = read_u64(f)

            activation = read_u32(f)

            neurons = []

            for neuron_index in range(
                n_neurons
            ):

                n_weights = read_u64(f)

                weights = np.empty(
                    n_weights,
                    dtype=np.float64
                )

                for i in range(
                    n_weights
                ):
                    weights[i] = read_double(f)

                bias = read_double(f)

                neurons.append(
                    Neuron(
                        weights,
                        bias
                    )
                )

            layers.append(
                Layer(
                    neurons,
                    activation
                )
            )

            if activation == Network.RELU:
                activation_name = "RELU"

            elif activation == Network.SOFTMAX:
                activation_name = "SOFTMAX"

            else:
                activation_name = (
                    f"UNKNOWN({activation})"
                )

            print(
                f"  Layer {layer_index}: "
                f"{n_neurons} neurons, "
                f"{activation_name}"
            )

        print(
            "Network loaded successfully.\n"
        )

        return Network(
            input_size,
            layers
        )


# ============================================================
# Loss
# ============================================================

def categorical_cross_entropy(
    label,
    prediction
):
    """
    Equivalent to:

        -log(P(correct_class))

    for a one-hot target.
    """

    probability = float(
        prediction[label]
    )

    # Prevent log(0).
    probability = max(
        probability,
        1e-15
    )

    return -math.log(
        probability
    )


# ============================================================
# Image saving
# ============================================================

def save_image(
    image,
    filename
):

    image = np.asarray(
        image,
        dtype=np.uint8
    )

    Image.fromarray(
        image,
        mode="L"
    ).save(filename)


# ============================================================
# Probability printing
# ============================================================

def print_probabilities(
    prediction
):

    print(
        "\nNetwork probabilities:"
    )

    for digit in range(
        NUM_CLASSES
    ):

        probability = float(
            prediction[digit]
        )

        print(
            f"  Digit {digit}: "
            f"{probability:.15e} "
            f"({probability * 100.0:.12e}%)"
        )


# ============================================================
# Prediction information
# ============================================================

def prediction_information(
    prediction,
    label
):

    predicted_label = int(
        np.argmax(prediction)
    )

    predicted_probability = float(
        prediction[predicted_label]
    )

    correct_probability = float(
        prediction[label]
    )

    loss = categorical_cross_entropy(
        label,
        prediction
    )

    return (
        predicted_label,
        predicted_probability,
        correct_probability,
        loss
    )


# ============================================================
# Network sanity check
# ============================================================

def sanity_check(
    network,
    images
):

    print(
        "========================================"
    )

    print(
        "NETWORK SANITY CHECK"
    )

    print(
        "========================================"
    )

    image = images[0]

    x = (
        image
        .astype(np.float64)
        .reshape(-1)
        / 255.0
    )

    prediction = network.forward(x)

    probability_sum = float(
        np.sum(prediction)
    )

    minimum = float(
        np.min(prediction)
    )

    maximum = float(
        np.max(prediction)
    )

    print(
        f"Probability sum:   "
        f"{probability_sum:.15f}"
    )

    print(
        f"Minimum output:    "
        f"{minimum:.15e}"
    )

    print(
        f"Maximum output:    "
        f"{maximum:.15e}"
    )

    if not np.all(
        np.isfinite(prediction)
    ):
        raise FloatingPointError(
            "Network produced NaN or Inf"
        )

    if abs(
        probability_sum - 1.0
    ) > 1e-10:

        print(
            "WARNING: Softmax probabilities "
            "do not sum to 1."
        )

    else:

        print(
            "Softmax check: PASS"
        )

    print()


# ============================================================
# MANUAL mode
# ============================================================

def manual_test(
    network,
    images,
    labels
):

    print(
        "========================================"
    )

    print(
        "MANUAL TEST MODE"
    )

    print(
        "========================================"
    )

    print(
        f"Images will be saved to:"
    )

    print(
        f"  {OUTPUT_DIR}/test.png"
    )

    print()

    print(
        "Press ENTER to continue."
    )

    print(
        "Enter q + ENTER to quit."
    )

    print()

    for index in range(
        len(images)
    ):

        image = images[index]

        label = int(
            labels[index]
        )

        # ----------------------------------------------------
        # Convert image to network input
        # ----------------------------------------------------

        x = (
            image
            .astype(np.float64)
            .reshape(-1)
            / 255.0
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = network.forward(x)

        (
            predicted_label,
            predicted_probability,
            correct_probability,
            loss
        ) = prediction_information(
            prediction,
            label
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        output_file = os.path.join(
            OUTPUT_DIR,
            "test.png"
        )

        save_image(
            image,
            output_file
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            "========================================"
        )

        print(
            f"Image:              "
            f"{index + 1}/{len(images)}"
        )

        print(
            f"True label:         "
            f"{label}"
        )

        print(
            f"Predicted label:    "
            f"{predicted_label}"
        )

        print(
            f"Prediction confidence: "
            f"{predicted_probability:.15e}"
        )

        print(
            f"Correct-class probability: "
            f"{correct_probability:.15e}"
        )

        print(
            f"Cross-entropy loss: "
            f"{loss:.15f}"
        )

        print(
            f"Correct:            "
            f"{'YES' if predicted_label == label else 'NO'}"
        )

        print_probabilities(
            prediction
        )

        print()

        print(
            f"Image saved to:"
        )

        print(
            f"  {output_file}"
        )

        # ----------------------------------------------------
        # Pause
        # ----------------------------------------------------

        try:

            command = input(
                "\nPress ENTER for next image "
                "(q + ENTER to quit): "
            )

            if command.strip().lower() == "q":

                print(
                    "\nStopping manual test."
                )

                break

        except KeyboardInterrupt:

            print(
                "\nStopping manual test."
            )

            break


# ============================================================
# SUMMARY mode
# ============================================================

def summary_test(
    network,
    images,
    labels
):

    print(
        "========================================"
    )

    print(
        "SUMMARY TEST MODE"
    )

    print(
        "========================================"
    )

    total = len(images)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_loss = 0.0

    n_correct = 0

    # --------------------------------------------------------
    # Confusion matrix
    #
    # confusion[actual][predicted]
    # --------------------------------------------------------

    confusion = np.zeros(
        (
            NUM_CLASSES,
            NUM_CLASSES
        ),
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Worst loss
    # --------------------------------------------------------

    worst_loss = -float("inf")

    worst_loss_index = -1

    worst_loss_prediction = None

    worst_loss_label = -1

    # --------------------------------------------------------
    # Worst incorrect confidence
    #
    # This finds the incorrectly classified image for which
    # the network is most confident in its WRONG prediction.
    # --------------------------------------------------------

    worst_confidence = -float("inf")

    worst_confidence_index = -1

    worst_confidence_prediction = None

    worst_confidence_label = -1

    # --------------------------------------------------------
    # Test every image
    # --------------------------------------------------------

    for index in range(total):

        image = images[index]

        label = int(
            labels[index]
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        x = (
            image
            .astype(np.float64)
            .reshape(-1)
            / 255.0
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        prediction = network.forward(x)

        # ----------------------------------------------------
        # Validate numerical output
        # ----------------------------------------------------

        if not np.all(
            np.isfinite(prediction)
        ):

            raise FloatingPointError(
                f"Network produced NaN/Inf "
                f"on image {index}"
            )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predicted_label = int(
            np.argmax(prediction)
        )

        predicted_probability = float(
            prediction[predicted_label]
        )

        loss = categorical_cross_entropy(
            label,
            prediction
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        total_loss += loss

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        if predicted_label == label:
            n_correct += 1

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        confusion[
            label,
            predicted_label
        ] += 1

        # ----------------------------------------------------
        # Worst loss
        # ----------------------------------------------------

        if loss > worst_loss:

            worst_loss = loss

            worst_loss_index = index

            worst_loss_prediction = (
                prediction.copy()
            )

            worst_loss_label = label

        # ----------------------------------------------------
        # Worst incorrect confidence
        # ----------------------------------------------------

        if predicted_label != label:

            if (
                predicted_probability
                > worst_confidence
            ):

                worst_confidence = (
                    predicted_probability
                )

                worst_confidence_index = (
                    index
                )

                worst_confidence_prediction = (
                    prediction.copy()
                )

                worst_confidence_label = (
                    label
                )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (index + 1) % 500 == 0
            or index + 1 == total
        ):

            print(
                f"Tested "
                f"{index + 1}/{total} images..."
            )

    # ========================================================
    # Final statistics
    # ========================================================

    average_loss = (
        total_loss / total
    )

    accuracy = (
        float(n_correct)
        / float(total)
    )

    n_incorrect = (
        total - n_correct
    )

    print()

    print(
        "========================================"
    )

    print(
        "FINAL RESULT"
    )

    print(
        "========================================"
    )

    print(
        f"Total images:      "
        f"{total}"
    )

    print(
        f"Correct:            "
        f"{n_correct}"
    )

    print(
        f"Incorrect:          "
        f"{n_incorrect}"
    )

    print(
        f"Average loss:       "
        f"{average_loss:.10f}"
    )

    print(
        f"Accuracy:           "
        f"{accuracy * 100.0:.4f}%"
    )

    # ========================================================
    # Per-class accuracy
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "PER-CLASS ACCURACY"
    )

    print(
        "========================================"
    )

    for digit in range(
        NUM_CLASSES
    ):

        actual_count = int(
            np.sum(
                confusion[digit]
            )
        )

        correct_count = int(
            confusion[
                digit,
                digit
            ]
        )

        if actual_count > 0:

            class_accuracy = (
                float(correct_count)
                / float(actual_count)
            )

        else:

            class_accuracy = 0.0

        print(
            f"Digit {digit}: "
            f"{correct_count:4d}/"
            f"{actual_count:4d} "
            f"= "
            f"{class_accuracy * 100.0:7.3f}%"
        )

    # ========================================================
    # Confusion matrix
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "CONFUSION MATRIX"
    )

    print(
        "========================================"
    )

    print(
        "Actual \\ Predicted"
    )

    print(
        "       "
        + "".join(
            f"{i:7d}"
            for i in range(NUM_CLASSES)
        )
    )

    for actual in range(
        NUM_CLASSES
    ):

        print(
            f"{actual:6d}"
            + "".join(
                f"{confusion[actual, predicted]:7d}"
                for predicted in range(
                    NUM_CLASSES
                )
            )
        )

    # ========================================================
    # Worst loss image
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "WORST CASE BY CROSS-ENTROPY LOSS"
    )

    print(
        "========================================"
    )

    worst_loss_predicted = int(
        np.argmax(
            worst_loss_prediction
        )
    )

    worst_loss_confidence = float(
        worst_loss_prediction[
            worst_loss_predicted
        ]
    )

    worst_loss_correct_probability = float(
        worst_loss_prediction[
            worst_loss_label
        ]
    )

    print(
        f"Image index:        "
        f"{worst_loss_index}"
    )

    print(
        f"True label:         "
        f"{worst_loss_label}"
    )

    print(
        f"Predicted label:    "
        f"{worst_loss_predicted}"
    )

    print(
        f"Correct probability:"
        f" {worst_loss_correct_probability:.15e}"
    )

    print(
        f"Prediction confidence:"
        f" {worst_loss_confidence:.15e}"
    )

    print(
        f"Loss:               "
        f"{worst_loss:.15f}"
    )

    print(
        f"Correct:            "
        f"{'YES' if worst_loss_predicted == worst_loss_label else 'NO'}"
    )

    print_probabilities(
        worst_loss_prediction
    )

    worst_loss_file = (
        WORST_LOSS_FILE
    )

    save_image(
        images[worst_loss_index],
        worst_loss_file
    )

    print()

    print(
        "Worst-loss image saved to:"
    )

    print(
        f"  {worst_loss_file}"
    )

    # ========================================================
    # Worst confident incorrect prediction
    # ========================================================

    if worst_confidence_index >= 0:

        print()

        print(
            "========================================"
        )

        print(
            "WORST CONFIDENT INCORRECT PREDICTION"
        )

        print(
            "========================================"
        )

        worst_confidence_predicted = int(
            np.argmax(
                worst_confidence_prediction
            )
        )

        correct_probability = float(
            worst_confidence_prediction[
                worst_confidence_label
            ]
        )

        print(
            f"Image index:        "
            f"{worst_confidence_index}"
        )

        print(
            f"True label:         "
            f"{worst_confidence_label}"
        )

        print(
            f"Predicted label:    "
            f"{worst_confidence_predicted}"
        )

        print(
            f"Wrong prediction probability:"
            f" {worst_confidence:.15e}"
        )

        print(
            f"Correct probability:"
            f" {correct_probability:.15e}"
        )

        print(
            f"Correct:            NO"
        )

        print_probabilities(
            worst_confidence_prediction
        )

        worst_confidence_file = (
            WORST_CONFIDENCE_FILE
        )

        save_image(
            images[worst_confidence_index],
            worst_confidence_file
        )

        print()

        print(
            "Worst-confident-error image saved to:"
        )

        print(
            f"  {worst_confidence_file}"
        )

    else:

        print()

        print(
            "No incorrectly classified "
            "images were found."
        )

    # ========================================================
    # Summary
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "TEST COMPLETE"
    )

    print(
        "========================================"
    )


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Command line
    # --------------------------------------------------------

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            f"  python3 {sys.argv[0]} MANUAL"
        )

        print(
            f"  python3 {sys.argv[0]} SUMMARY"
        )

        sys.exit(1)

    mode = sys.argv[1].upper()

    if mode not in (
        "MANUAL",
        "SUMMARY"
    ):

        print(
            "Error: mode must be MANUAL or SUMMARY"
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Prepare directories
    # --------------------------------------------------------

    ensure_directories()

    # --------------------------------------------------------
    # Load MNIST
    # --------------------------------------------------------

    print(
        "========================================"
    )

    print(
        "Loading MNIST test set"
    )

    print(
        "========================================"
    )

    images, labels = load_mnist()

    # --------------------------------------------------------
    # Load network
    # --------------------------------------------------------

    print()

    print(
        "========================================"
    )

    print(
        "Loading neural network"
    )

    print(
        "========================================"
    )

    network = load_network(
        NETWORK_FILE
    )

    # --------------------------------------------------------
    # Validate architecture
    # --------------------------------------------------------

    if network.input_size != INPUT_SIZE:

        raise ValueError(
            f"Network input size is "
            f"{network.input_size}, "
            f"but MNIST requires "
            f"{INPUT_SIZE}"
        )

    if len(network.layers) == 0:

        raise ValueError(
            "Network contains no layers"
        )

    last_layer = network.layers[-1]

    if len(last_layer.neurons) != OUTPUT_SIZE:

        raise ValueError(
            f"Network output size is "
            f"{len(last_layer.neurons)}, "
            f"but expected "
            f"{OUTPUT_SIZE}"
        )

    if last_layer.activation != Network.SOFTMAX:

        raise ValueError(
            "Final network layer is not SOFTMAX"
        )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------

    sanity_check(
        network,
        images
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    if mode == "MANUAL":

        manual_test(
            network,
            images,
            labels
        )

    else:

        summary_test(
            network,
            images,
            labels
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()