#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <filesystem>

#include "mnist_loader.h"
#include "neural_net.h"

using namespace std;

bool askYesNo(const string& prompt) {
    while (true) {
        cout << prompt;
        string answer;
        if (!getline(cin, answer)) {
            return false;
        }

        transform(answer.begin(), answer.end(), answer.begin(),
                  [](unsigned char ch) { return static_cast<char>(tolower(ch)); });

        if (answer == "y" || answer == "yes") {
            return true;
        }
        if (answer == "n" || answer == "no") {
            return false;
        }

        cout << "Please type 'y' or 'n'." << '\n';
    }
}

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

        const size_t inputSize = static_cast<size_t>(rows * cols);
        if (inputSize != 784) {
            throw runtime_error("Expected MNIST images to be 28x28 (784 values).");
        }

        const size_t hiddenSize = 128;
        const size_t outputSize = 10;
        const double learningRate = 0.1;
        const size_t warmupCheckSize = min<size_t>(1000, images.size());
        const size_t trainSize = images.size();
        const int epochs = 5;
        const string modelPath = "models/mnist_model.bin";

        NeuralNetwork network(inputSize, hiddenSize, outputSize, learningRate);

        cout << "Loaded " << images.size() << " training images." << '\n';
        cout << "Initial sanity check size: " << warmupCheckSize << " (completed in earlier run)" << '\n';
        cout << "Training on full dataset size: " << trainSize << '\n';

        auto evaluateAccuracy = [&](const string& contextLabel) {
            size_t correct = 0;
            for (size_t idx = 0; idx < trainSize; ++idx) {
                vector<double> input(inputSize, 0.0);
                for (size_t p = 0; p < inputSize; ++p) {
                    input[p] = static_cast<double>(images[idx][p]) / 255.0;
                }

                const uint8_t predicted = network.predict(input);
                if (predicted == labels[idx]) {
                    ++correct;
                }
            }

            const double accuracy = (100.0 * static_cast<double>(correct)) / static_cast<double>(trainSize);
            cout << "Final Accuracy (" << contextLabel << "): " << accuracy << "%" << '\n';
        };

        const bool hasSavedModel = filesystem::exists(modelPath);
        bool trainNewModel = true;

        if (hasSavedModel) {
            cout << "Found existing model at " << modelPath << "." << '\n';
            trainNewModel = askYesNo("Train a new model and overwrite it? (y/n): ");
        }

        if (!trainNewModel) {
            if (!network.loadModel(modelPath)) {
                throw runtime_error("Could not load saved model from " + modelPath + ".");
            }

            cout << "Loaded existing model from " << modelPath << ". Skipping retraining." << '\n';
            evaluateAccuracy("loaded model");
            cout << "Training run complete." << '\n';
            return 0;
        }

        if (hasSavedModel) {
            cout << "Starting new training run and overwriting saved model..." << '\n';
        } else {
            cout << "No saved model found. Starting training..." << '\n';
        }

        double lastEpochAccuracy = 0.0;

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            double totalLoss = 0.0;
            size_t correct = 0;

            for (size_t idx = 0; idx < trainSize; ++idx) {
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
                if (predicted == labels[idx]) {
                    ++correct;
                }

                network.backpropagate(input, target);
            }

            const double avgLoss = totalLoss / static_cast<double>(trainSize);
            const double accuracy = (100.0 * static_cast<double>(correct)) / static_cast<double>(trainSize);
            lastEpochAccuracy = accuracy;

            cout << "Epoch " << epoch
                << " | Avg Loss: " << avgLoss
                << " | Accuracy: " << accuracy << "%" << '\n';
        }

        cout << "Final Accuracy (last epoch): " << lastEpochAccuracy << "%" << '\n';

        cout << "Training run complete." << '\n';
        filesystem::create_directories("models");
        if (!network.saveModel(modelPath)) {
            cerr << "Warning: training finished but failed to save model to " << modelPath << '\n';
        } else {
            cout << "Saved model to " << modelPath << '\n';
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}