#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <cstdint>
#include <string>
#include <vector>

class NeuralNetwork {
public:
	NeuralNetwork(std::size_t inputSize, std::size_t hiddenSize, std::size_t outputSize, double learningRate);

	std::vector<double> forward(const std::vector<double>& input);
	void backpropagate(const std::vector<double>& input, const std::vector<double>& target);
	uint8_t predict(const std::vector<double>& input);
	bool saveModel(const std::string& filePath) const;
	bool loadModel(const std::string& filePath);

private:
	static double sigmoid(double x);
	static double sigmoidDerivativeFromActivation(double activation);

	std::size_t inputSize_;
	std::size_t hiddenSize_;
	std::size_t outputSize_;
	double learningRate_;

	std::vector<std::vector<double>> weightsInputHidden_;
	std::vector<std::vector<double>> weightsHiddenOutput_;
	std::vector<double> biasHidden_;
	std::vector<double> biasOutput_;

	std::vector<double> lastHiddenActivations_;
	std::vector<double> lastOutputActivations_;
};

#endif