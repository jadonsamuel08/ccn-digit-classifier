#include "neural_network.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>

using namespace std;

NeuralNetwork::NeuralNetwork(size_t inputSize, size_t hiddenSize, size_t outputSize, double learningRate)
	: inputSize_(inputSize),
	  hiddenSize_(hiddenSize),
	  outputSize_(outputSize),
	  learningRate_(learningRate),
	  weightsInputHidden_(inputSize, vector<double>(hiddenSize, 0.0)),
	  weightsHiddenOutput_(hiddenSize, vector<double>(outputSize, 0.0)),
	  biasHidden_(hiddenSize, 0.0),
	  biasOutput_(outputSize, 0.0),
	  lastHiddenActivations_(hiddenSize, 0.0),
	  lastOutputActivations_(outputSize, 0.0) {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> dist(-0.5, 0.5);

	for (size_t i = 0; i < inputSize_; ++i) {
		for (size_t j = 0; j < hiddenSize_; ++j) {
			weightsInputHidden_[i][j] = dist(gen);
		}
	}

	for (size_t j = 0; j < hiddenSize_; ++j) {
		for (size_t k = 0; k < outputSize_; ++k) {
			weightsHiddenOutput_[j][k] = dist(gen);
		}
	}

	for (double& value : biasHidden_) {
		value = dist(gen);
	}

	for (double& value : biasOutput_) {
		value = dist(gen);
	}
}

double NeuralNetwork::sigmoid(double x) {
	return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::sigmoidDerivativeFromActivation(double activation) {
	return activation * (1.0 - activation);
}

vector<double> NeuralNetwork::forward(const vector<double>& input) {
	if (input.size() != inputSize_) {
		throw runtime_error("Input size does not match network input layer size.");
	}

	for (size_t j = 0; j < hiddenSize_; ++j) {
		double sum = biasHidden_[j];
		for (size_t i = 0; i < inputSize_; ++i) {
			sum += input[i] * weightsInputHidden_[i][j];
		}
		lastHiddenActivations_[j] = sigmoid(sum);
	}

	for (size_t k = 0; k < outputSize_; ++k) {
		double sum = biasOutput_[k];
		for (size_t j = 0; j < hiddenSize_; ++j) {
			sum += lastHiddenActivations_[j] * weightsHiddenOutput_[j][k];
		}
		lastOutputActivations_[k] = sigmoid(sum);
	}

	return lastOutputActivations_;
}

void NeuralNetwork::backpropagate(const vector<double>& input, const vector<double>& target) {
	if (target.size() != outputSize_) {
		throw runtime_error("Target size does not match network output layer size.");
	}

	forward(input);

	vector<double> outputDelta(outputSize_, 0.0);
	for (size_t k = 0; k < outputSize_; ++k) {
		const double error = lastOutputActivations_[k] - target[k];
		outputDelta[k] = error * sigmoidDerivativeFromActivation(lastOutputActivations_[k]);
	}

	vector<double> hiddenDelta(hiddenSize_, 0.0);
	for (size_t j = 0; j < hiddenSize_; ++j) {
		double propagatedError = 0.0;
		for (size_t k = 0; k < outputSize_; ++k) {
			propagatedError += outputDelta[k] * weightsHiddenOutput_[j][k];
		}
		hiddenDelta[j] = propagatedError * sigmoidDerivativeFromActivation(lastHiddenActivations_[j]);
	}

	for (size_t j = 0; j < hiddenSize_; ++j) {
		for (size_t k = 0; k < outputSize_; ++k) {
			weightsHiddenOutput_[j][k] -= learningRate_ * lastHiddenActivations_[j] * outputDelta[k];
		}
	}

	for (size_t k = 0; k < outputSize_; ++k) {
		biasOutput_[k] -= learningRate_ * outputDelta[k];
	}

	for (size_t i = 0; i < inputSize_; ++i) {
		for (size_t j = 0; j < hiddenSize_; ++j) {
			weightsInputHidden_[i][j] -= learningRate_ * input[i] * hiddenDelta[j];
		}
	}

	for (size_t j = 0; j < hiddenSize_; ++j) {
		biasHidden_[j] -= learningRate_ * hiddenDelta[j];
	}
}

uint8_t NeuralNetwork::predict(const vector<double>& input) {
	const vector<double> output = forward(input);
	const auto best = max_element(output.begin(), output.end());
	return static_cast<uint8_t>(distance(output.begin(), best));
}

bool NeuralNetwork::saveModel(const string& filePath) const {
	ofstream out(filePath, ios::binary);
	if (!out.is_open()) {
		return false;
	}

	const uint32_t magic = 0x4D4E4953; // MNIS
	const uint64_t in = static_cast<uint64_t>(inputSize_);
	const uint64_t hidden = static_cast<uint64_t>(hiddenSize_);
	const uint64_t outSize = static_cast<uint64_t>(outputSize_);

	out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
	out.write(reinterpret_cast<const char*>(&in), sizeof(in));
	out.write(reinterpret_cast<const char*>(&hidden), sizeof(hidden));
	out.write(reinterpret_cast<const char*>(&outSize), sizeof(outSize));

	for (size_t i = 0; i < inputSize_; ++i) {
		out.write(reinterpret_cast<const char*>(weightsInputHidden_[i].data()),
			static_cast<streamsize>(hiddenSize_ * sizeof(double)));
	}

	for (size_t j = 0; j < hiddenSize_; ++j) {
		out.write(reinterpret_cast<const char*>(weightsHiddenOutput_[j].data()),
			static_cast<streamsize>(outputSize_ * sizeof(double)));
	}

	out.write(reinterpret_cast<const char*>(biasHidden_.data()),
		static_cast<streamsize>(hiddenSize_ * sizeof(double)));
	out.write(reinterpret_cast<const char*>(biasOutput_.data()),
		static_cast<streamsize>(outputSize_ * sizeof(double)));

	return out.good();
}

bool NeuralNetwork::loadModel(const string& filePath) {
	ifstream in(filePath, ios::binary);
	if (!in.is_open()) {
		return false;
	}

	uint32_t magic = 0;
	uint64_t inSize = 0;
	uint64_t hidden = 0;
	uint64_t outSize = 0;

	in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
	in.read(reinterpret_cast<char*>(&inSize), sizeof(inSize));
	in.read(reinterpret_cast<char*>(&hidden), sizeof(hidden));
	in.read(reinterpret_cast<char*>(&outSize), sizeof(outSize));

	if (!in.good() ||
		magic != 0x4D4E4953 ||
		inSize != static_cast<uint64_t>(inputSize_) ||
		hidden != static_cast<uint64_t>(hiddenSize_) ||
		outSize != static_cast<uint64_t>(outputSize_)) {
		return false;
	}

	for (size_t i = 0; i < inputSize_; ++i) {
		in.read(reinterpret_cast<char*>(weightsInputHidden_[i].data()),
			static_cast<streamsize>(hiddenSize_ * sizeof(double)));
	}

	for (size_t j = 0; j < hiddenSize_; ++j) {
		in.read(reinterpret_cast<char*>(weightsHiddenOutput_[j].data()),
			static_cast<streamsize>(outputSize_ * sizeof(double)));
	}

	in.read(reinterpret_cast<char*>(biasHidden_.data()),
		static_cast<streamsize>(hiddenSize_ * sizeof(double)));
	in.read(reinterpret_cast<char*>(biasOutput_.data()),
		static_cast<streamsize>(outputSize_ * sizeof(double)));

	return in.good();
}