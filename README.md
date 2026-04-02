# MNIST Digit Classifier (From Scratch in C++)

This is a beginner-friendly neural network project built in C++.

The goal is to learn how a neural network works under the hood, without using any machine learning libraries.

## What this project does

- Loads MNIST handwritten digit images and labels
- Builds a small neural network from scratch
- First verified training on a small subset (1000 images)
- Now trains on the full training dataset
- Uses forward propagation + backpropagation
- Prints loss and accuracy each epoch

## Network shape

This model has 3 layers:

- Input layer: 784 neurons (28x28 image pixels flattened)
- Hidden layer: 128 neurons
- Output layer: 10 neurons (digits 0-9)

Weight matrix sizes:

- Input to hidden: `784 x 128`
- Hidden to output: `128 x 10`

## How training works (simple version)

### 1) Initialize weights

All weights are initialized randomly between `-0.5` and `0.5`.

This helps break symmetry so neurons can learn different things.

### 2) Forward propagation

For one input image:

1. Multiply input vector by first weight matrix
2. Add hidden bias
3. Apply sigmoid activation
4. Multiply hidden activations by second weight matrix
5. Add output bias
6. Apply sigmoid activation

The result is 10 output values (one score per digit class).

### 3) Compute error

Target is one-hot encoded, for example digit `3`:

`[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`

Error is based on difference between prediction and target.

### 4) Backpropagation

- Compute output layer deltas
- Propagate error backward to hidden layer deltas
- Compute gradients
- Update weights and biases using gradient descent

Update rule concept:

`new_weight = old_weight - learning_rate * gradient`

## Build and run

From the project root:

```bash
make clean
make
make run
```

To evaluate on the MNIST test set (the remaining data):

```bash
make test
```

## Latest results

- Full training run: 5 epochs on 60,000 training images
- Test set evaluation: 10,000 images from `t10k-*` files
- Observed test accuracy: **96.64%**
- Observed test average loss: **0.0310317**

This is a strong result for a first from-scratch network and confirms that the model generalizes well beyond the training data.

## Current training setup

- Learning rate: `0.1`
- Epochs: `5`
- Initial verification run: `1000` images
- Current run: full training set (`60000` images from `train-*` files)
- Activation: sigmoid
- Loss: mean squared error style accumulation (`0.5 * (output - target)^2`)

The project now also includes a separate test evaluator in `src/evaluate_test.cpp` that loads the saved model and reports accuracy on the test set.

Starting with 1000 images was used as a beginner-friendly sanity check. After confirming forward and backprop worked, training was scaled to the full dataset.

## Project structure

- `src/main.cpp`: training loop and data preparation
- `src/mnist_loader.cpp`: reads MNIST binary files
- `src/neural_network.cpp`: forward pass and backprop logic
- `include/neural_network.h`: neural network class definition
- `archive/`: MNIST data files

## Future ideas

After this works reliably, good next steps are:

- Shuffle training data each epoch
- Train with mini-batches
- Try ReLU + softmax later for better performance
