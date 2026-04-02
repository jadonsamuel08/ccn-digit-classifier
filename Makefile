CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude
TARGET := mnist_viewer
TEST_TARGET := mnist_test
SRCS := src/main.cpp src/mnist_loader.cpp src/neural_network.cpp
TEST_SRCS := src/evaluate_test.cpp src/mnist_loader.cpp src/neural_network.cpp

.PHONY: all run test clean

all: $(TARGET) $(TEST_TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

$(TEST_TARGET): $(TEST_SRCS)
	$(CXX) $(CXXFLAGS) $(TEST_SRCS) -o $(TEST_TARGET)

run: $(TARGET)
	./$(TARGET)

test: $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -f $(TARGET) $(TEST_TARGET)
