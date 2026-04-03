#include "mnist_loader.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

// Reads each line 4 bytes at a time and converts to uint32_t
uint32_t readBigEndianUInt32(ifstream& inFile) {
	unsigned char bytes[4];
	inFile.read(reinterpret_cast<char*>(bytes), 4);
	if (!inFile) {
		throw runtime_error("Failed to read 4-byte header value.");
	}

	return (static_cast<uint32_t>(bytes[0]) << 24) | // | --> Adds up binary for each byte of data to form a single number
		    (static_cast<uint32_t>(bytes[1]) << 16) |
		    (static_cast<uint32_t>(bytes[2]) << 8) |
		    static_cast<uint32_t>(bytes[3]);
}

// Loads Labels
vector<uint8_t> loadMnistLabels(const string& path) {
	ifstream inFile(path, ios::binary);
	if (!inFile) {
		throw runtime_error("Could not open labels file: " + path);
	}

	const uint32_t magic = readBigEndianUInt32(inFile); // Tells the program that the file is a label.
	const uint32_t numLabels = readBigEndianUInt32(inFile); // Number of Labels

	if (magic != 2049) {
		throw runtime_error("Invalid labels magic number. Expected 2049.");
	}

	vector<uint8_t> labels(numLabels);
	inFile.read(reinterpret_cast<char*>(labels.data()), static_cast<streamsize>(numLabels));
	if (!inFile) {
		throw runtime_error("Failed to read all labels bytes.");
	}

	return labels;
}

// Loads Images
vector<vector<uint8_t>> loadMnistImages(const string& path, uint32_t& rows, uint32_t& cols) {
	ifstream inFile(path, ios::binary);
	if (!inFile) {
		throw runtime_error("Could not open images file: " + path);
	}

	const uint32_t magic = readBigEndianUInt32(inFile); // Tells the program that the file is a image.
	const uint32_t numImages = readBigEndianUInt32(inFile); // Number of Images
	rows = readBigEndianUInt32(inFile); // 28 pixels tall
	cols = readBigEndianUInt32(inFile); // 28 pixels wide

	if (magic != 2051) {
		throw runtime_error("Invalid images magic number. Expected 2051.");
	}

	const uint32_t imageSize = rows * cols;
	vector<vector<uint8_t>> images(numImages, vector<uint8_t>(imageSize));

	for (uint32_t i = 0; i < numImages; ++i) {
		inFile.read(reinterpret_cast<char*>(images[i].data()), static_cast<streamsize>(imageSize));
		if (!inFile) {
			throw runtime_error("Failed to read image bytes at index " + to_string(i));
		}
	}

	return images;
}

// Prints image with hashtags
void printImageAscii(const vector<uint8_t>& image, uint32_t rows, uint32_t cols) {
	for (uint32_t r = 0; r < rows; ++r) {
		for (uint32_t c = 0; c < cols; ++c) {
			const uint8_t pixel = image[r * cols + c];
            // if (pixel > 128) {cout << "##";}
            // else {cout << "  ";}
			cout << (pixel > 128 ? "##" : "  ");
		}
		cout << '\n';
	}
}
