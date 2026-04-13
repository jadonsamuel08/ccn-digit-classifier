#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mnist_loader.h"
#include "neural_net.h"

using namespace std;

int main() {
	try {
		const string testLabelsPath = "data/t10k-labels.idx1-ubyte";
		const string testImagesPath = "data/t10k-images.idx3-ubyte";
		const string modelPath = "models/mnist_model.bin";

		uint32_t rows = 0;
		uint32_t cols = 0;

		vector<uint8_t> labels = loadMnistLabels(testLabelsPath);
		vector<vector<uint8_t>> images = loadMnistImages(testImagesPath, rows, cols);

		if (labels.size() != images.size()) {
			throw runtime_error("Test labels count does not match test images count.");
		}

		const size_t inputSize = static_cast<size_t>(rows * cols);
		if (inputSize != 784) {
			throw runtime_error("Expected MNIST test images to be 28x28 (784 values).");
		}

		const size_t hiddenSize = 128;
		const size_t outputSize = 10;
		const double learningRate = 0.1;
		const size_t testSize = images.size();

		NeuralNetwork network(inputSize, hiddenSize, outputSize, learningRate);
		if (!network.loadModel(modelPath)) {
			throw runtime_error("Could not load saved model from " + modelPath + ". Run training first.");
		}

		double totalLoss = 0.0;
		size_t correct = 0;
		array<size_t, 10> totalByLabel{};
		array<size_t, 10> correctByLabel{};
		array<array<size_t, 10>, 10> confusion{};

		for (size_t idx = 0; idx < testSize; ++idx) {
			vector<double> input(inputSize, 0.0);
			for (size_t p = 0; p < inputSize; ++p) {
				input[p] = static_cast<double>(images[idx][p]) / 255.0;
			}

			vector<double> target(outputSize, 0.0);
			target[labels[idx]] = 1.0;

			const vector<double> output = network.forward(input);

			for (size_t k = 0; k < outputSize; ++k) {
				const double diff = output[k] - target[k];
				totalLoss += 0.5 * diff * diff;
			}

			const auto best = max_element(output.begin(), output.end());
			const uint8_t predicted = static_cast<uint8_t>(distance(output.begin(), best));
			const uint8_t actual = labels[idx];
			++totalByLabel[actual];
			++confusion[actual][predicted];
			if (predicted == labels[idx]) {
				++correct;
				++correctByLabel[actual];
			}
		}

		const double avgLoss = totalLoss / static_cast<double>(testSize);
		const double accuracy = (100.0 * static_cast<double>(correct)) / static_cast<double>(testSize);

		cout << "Loaded model from " << modelPath << '\n';
		cout << "Evaluated on " << testSize << " MNIST test images.\n";
		cout << "Test Avg Loss: " << avgLoss << '\n';
		cout << "Test Accuracy: " << accuracy << "%\n";

		cout << "\nPer-digit accuracy:\n";
		for (int digit = 0; digit < 10; ++digit) {
			double digitAccuracy = 0.0;
			if (totalByLabel[digit] > 0) {
				digitAccuracy =
					(100.0 * static_cast<double>(correctByLabel[digit])) /
					static_cast<double>(totalByLabel[digit]);
			}

			cout << "Digit " << digit
				 << " | Correct: " << correctByLabel[digit]
				 << "/" << totalByLabel[digit]
				 << " | Accuracy: " << fixed << setprecision(2)
				 << digitAccuracy << "%\n";
		}

		cout << "\nConfusion matrix (rows=actual, cols=predicted):\n";
		cout << "      ";
		for (int col = 0; col < 10; ++col) {
			cout << setw(6) << col;
		}
		cout << '\n';

		for (int row = 0; row < 10; ++row) {
			cout << "A" << row << " ->";
			for (int col = 0; col < 10; ++col) {
				cout << setw(6) << confusion[row][col];
			}
			cout << '\n';
		}
	} catch (const exception& e) {
		cerr << "Error: " << e.what() << '\n';
		return 1;
	}

	return 0;
}