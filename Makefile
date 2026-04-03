CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude

BUILD_DIR := build
PREVIEW_BIN := $(BUILD_DIR)/preview
TRAIN_BIN := $(BUILD_DIR)/train
TEST_BIN := $(BUILD_DIR)/test

PREVIEW_SRCS := src/preview.cpp src/mnist_loader.cpp
TRAIN_SRCS := src/train.cpp src/mnist_loader.cpp src/neural_net.cpp
TEST_SRCS := src/test.cpp src/mnist_loader.cpp src/neural_net.cpp

.PHONY: all run preview train test clean

all: $(PREVIEW_BIN) $(TRAIN_BIN) $(TEST_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(PREVIEW_BIN): $(PREVIEW_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(PREVIEW_SRCS) -o $(PREVIEW_BIN)

$(TRAIN_BIN): $(TRAIN_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TRAIN_SRCS) -o $(TRAIN_BIN)

$(TEST_BIN): $(TEST_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_SRCS) -o $(TEST_BIN)

run: preview

preview: $(PREVIEW_BIN)
	./$(PREVIEW_BIN)

train: $(TRAIN_BIN)
	./$(TRAIN_BIN)

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -rf $(BUILD_DIR) preview train test .preview.stamp .train.stamp .test.stamp
