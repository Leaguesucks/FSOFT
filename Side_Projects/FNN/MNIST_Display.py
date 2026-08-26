import sys
import numpy as np
import matplotlib.pyplot as plt


def display_image(filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()

    label = int(lines[0].strip())

    pixels = []

    for line in lines[1:]:
        pixels.extend(float(x) for x in line.split())

    if len(pixels) != 28 * 28:
        raise ValueError(
            f"Expected 784 pixels, got {len(pixels)}"
        )

    image = np.array(pixels).reshape(28, 28)

    plt.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0
    )

    plt.title(f"Label: {label}")
    plt.axis("off")

    # Save instead of displaying
    output_file = ".out/mnist.png"
    plt.savefig(
        output_file,
        bbox_inches="tight",
        pad_inches=0
    )

    print(f"Image saved to {output_file}")
    print(f"Label: {label}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <image_file>")
        sys.exit(1)

    display_image(sys.argv[1])