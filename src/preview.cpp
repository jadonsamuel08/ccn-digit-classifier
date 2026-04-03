#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mnist_loader.h"

using namespace std;

int main() {
	try {
		const string labelsPath = "data/train-labels.idx1-ubyte";
		const string imagesPath = "data/train-images.idx3-ubyte";

		uint32_t rows = 0;
		uint32_t cols = 0;

		vector<uint8_t> labels = loadMnistLabels(labelsPath);
		vector<vector<uint8_t>> images = loadMnistImages(imagesPath, rows, cols);

		if (labels.size() != images.size()) {
			throw runtime_error("Labels count does not match images count.");
		}

		if (rows * cols != 784) {
			throw runtime_error("Expected MNIST images to be 28x28 (784 values).");
		}

		cout << "Loaded " << images.size() << " training images." << '\n';
		cout << "Each image size: " << rows << " x " << cols << '\n';

		while (true) {
			cout << "\nEnter image index (0 to 59999, -1 to quit): ";

			int index = -1;
			cin >> index;

			if (!cin) {
				cin.clear();
				cin.ignore(numeric_limits<streamsize>::max(), '\n');
				cout << "Please enter a number." << '\n';
				continue;
			}

			if (index == -1) {
				cout << "Goodbye!" << '\n';
				break;
			}

			if (index < 0 || static_cast<size_t>(index) >= images.size()) {
				cout << "That index is out of range." << '\n';
				continue;
			}

			cout << "Label at index " << index << ": "
				<< static_cast<int>(labels[static_cast<size_t>(index)]) << '\n';
			cout << "Image preview:" << '\n';
			printImageAscii(images[static_cast<size_t>(index)], rows, cols);
		}
	} catch (const exception& e) {
		cerr << "Error: " << e.what() << '\n';
		return 1;
	}

	return 0;
}