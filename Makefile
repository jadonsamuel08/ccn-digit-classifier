CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude -I/opt/homebrew/opt/raylib/include
RAYLIB_FLAGS := -L/opt/homebrew/opt/raylib/lib -lraylib -lm

BUILD_DIR := build
PREVIEW_BIN := $(BUILD_DIR)/preview
TRAIN_BIN := $(BUILD_DIR)/train
TEST_BIN := $(BUILD_DIR)/test
GUI_BIN := $(BUILD_DIR)/gui

PREVIEW_SRCS := src/preview.cpp src/mnist_loader.cpp
TRAIN_SRCS := src/train.cpp src/mnist_loader.cpp src/neural_net.cpp
TEST_SRCS := src/test.cpp src/mnist_loader.cpp src/neural_net.cpp
GUI_SRCS := src/draw_gui.cpp src/mnist_loader.cpp src/neural_net.cpp

.PHONY: all run preview train test gui clean

all: $(PREVIEW_BIN) $(TRAIN_BIN) $(TEST_BIN) $(GUI_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(PREVIEW_BIN): $(PREVIEW_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(PREVIEW_SRCS) -o $(PREVIEW_BIN)

$(TRAIN_BIN): $(TRAIN_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TRAIN_SRCS) -o $(TRAIN_BIN)

$(TEST_BIN): $(TEST_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(TEST_SRCS) -o $(TEST_BIN)

$(GUI_BIN): $(GUI_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(GUI_SRCS) -o $(GUI_BIN) $(RAYLIB_FLAGS)

run: preview

preview: $(PREVIEW_BIN)
	./$(PREVIEW_BIN)

train: $(TRAIN_BIN)
	./$(TRAIN_BIN)

test: $(TEST_BIN)
	./$(TEST_BIN)

gui: $(GUI_BIN)
	./$(GUI_BIN)

clean:
	rm -rf $(BUILD_DIR) preview train test .preview.stamp .train.stamp .test.stamp
