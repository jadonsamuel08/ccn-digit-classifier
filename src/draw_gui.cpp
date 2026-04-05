#include <raylib.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <iostream>

#include "neural_net.h"
#include "mnist_loader.h"

// Constants for the GUI
const int CANVAS_SIZE = 280;
const int DISPLAY_SCALE = 1;  // Display the canvas at 1x scale to fit in 600x400
const int DISPLAY_WIDTH = CANVAS_SIZE * DISPLAY_SCALE;
const int DISPLAY_HEIGHT = CANVAS_SIZE * DISPLAY_SCALE;
const int BRUSH_SIZE = 10;

// Drawing canvas (280x280)
std::vector<std::vector<uint8_t>> canvas;

// Function to downscale 280x280 to 28x28
std::vector<std::vector<uint8_t>> downscaleCanvas(const std::vector<std::vector<uint8_t>>& srcCanvas) {
    const int srcSize = 280;
    const int dstSize = 28;
    const int scale = srcSize / dstSize;  // Should be 10
    
    std::vector<std::vector<uint8_t>> result(dstSize, std::vector<uint8_t>(dstSize, 0));
    
    for (int y = 0; y < dstSize; y++) {
        for (int x = 0; x < dstSize; x++) {
            // Calculate average of the 10x10 region
            int sum = 0;
            int count = 0;
            
            for (int dy = 0; dy < scale; dy++) {
                for (int dx = 0; dx < scale; dx++) {
                    int srcY = y * scale + dy;
                    int srcX = x * scale + dx;
                    if (srcY < srcSize && srcX < srcSize) {
                        sum += srcCanvas[srcY][srcX];
                        count++;
                    }
                }
            }
            
            if (count > 0) {
                result[y][x] = static_cast<uint8_t>(sum / count);
            }
        }
    }
    
    return result;
}

// Convert 28x28 canvas to vector of normalized doubles for neural network
std::vector<double> canvasToNeuralNetInput(const std::vector<std::vector<uint8_t>>& downscaledCanvas) {
    std::vector<double> input;
    input.reserve(28 * 28);
    
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            // Normalize to 0.0-1.0 range
            double normalized = downscaledCanvas[y][x] / 255.0;
            input.push_back(normalized);
        }
    }
    
    return input;
}

// Draw a circular brush on the canvas
void drawBrushStroke(int centerX, int centerY) {
    int radius = BRUSH_SIZE;
    
    // Convert from display coordinates to canvas coordinates
    int canvasX = centerX / DISPLAY_SCALE;
    int canvasY = centerY / DISPLAY_SCALE;
    
    // Draw filled circle on canvas
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int y = canvasY + dy;
            int x = canvasX + dx;
            
            // Check if within circle and within canvas bounds
            if (dx*dx + dy*dy <= radius*radius && 
                x >= 0 && x < CANVAS_SIZE && 
                y >= 0 && y < CANVAS_SIZE) {
                canvas[y][x] = 255;  // White
            }
        }
    }
}

// Render the canvas to the screen at 4x scale
void renderCanvas(int offsetX, int offsetY) {
    for (int y = 0; y < CANVAS_SIZE; y++) {
        for (int x = 0; x < CANVAS_SIZE; x++) {
            uint8_t pixelValue = canvas[y][x];
            Color pixelColor = {pixelValue, pixelValue, pixelValue, 255};
            
            // Draw scaled pixels (4x4 each)
            Rectangle dest = {
                static_cast<float>(offsetX + x * DISPLAY_SCALE),
                static_cast<float>(offsetY + y * DISPLAY_SCALE),
                static_cast<float>(DISPLAY_SCALE),
                static_cast<float>(DISPLAY_SCALE)
            };
            DrawRectangleRec(dest, pixelColor);
        }
    }
    
    // Draw border around the canvas
    DrawRectangleLines(offsetX, offsetY, DISPLAY_WIDTH, DISPLAY_HEIGHT, BLACK);
}

// Clear the canvas
void clearCanvas() {
    for (int y = 0; y < CANVAS_SIZE; y++) {
        for (int x = 0; x < CANVAS_SIZE; x++) {
            canvas[y][x] = 0;  // Black
        }
    }
}

int main() {
    // Initialize canvas with zeros (black)
    canvas = std::vector<std::vector<uint8_t>>(CANVAS_SIZE, std::vector<uint8_t>(CANVAS_SIZE, 0));
    
    // Load the neural network model
    NeuralNetwork net(784, 128, 10, 0.1);
    if (!net.loadModel("models/mnist_model.bin")) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }
    
    // Initialize raylib window
    const int screenWidth = 600;
    const int screenHeight = 400;
    InitWindow(screenWidth, screenHeight, "Digit Recognizer - Draw to Predict");
    SetTargetFPS(60);
    
    // Button state
    bool predictPressed = false;
    std::vector<double> lastPredictionConfidences;
    uint8_t lastPredictedDigit = 10;  // Invalid initially
    
    // Layout parameters
    const int canvasOffsetX = 10;
    const int canvasOffsetY = 10;
    const int buttonWidth = 100;
    const int buttonHeight = 40;
    const int predictButtonX = canvasOffsetX + DISPLAY_WIDTH + 15;
    const int predictButtonY = canvasOffsetY;
    const int clearButtonX = predictButtonX;
    const int clearButtonY = predictButtonY + buttonHeight + 10;
    
    // Prediction display area
    const int predictionX = predictButtonX;
    const int predictionY = clearButtonY + buttonHeight + 10;
    
    // Main loop
    while (!WindowShouldClose()) {
        // Handle mouse drawing
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            int mouseX = GetMouseX();
            int mouseY = GetMouseY();
            
            // Check if mouse is within canvas bounds
            if (mouseX >= canvasOffsetX && mouseX < canvasOffsetX + DISPLAY_WIDTH &&
                mouseY >= canvasOffsetY && mouseY < canvasOffsetY + DISPLAY_HEIGHT) {
                
                int relativeX = mouseX - canvasOffsetX;
                int relativeY = mouseY - canvasOffsetY;
                drawBrushStroke(relativeX, relativeY);
            }
        }
        
        // Handle predict button
        Rectangle predictButtonRect = {(float)predictButtonX, (float)predictButtonY, 
                                       (float)buttonWidth, (float)buttonHeight};
        bool predictHovered = CheckCollisionPointRec(GetMousePosition(), predictButtonRect);
        
        if (predictHovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            // Downscale canvas
            auto downscaled = downscaleCanvas(canvas);
            
            // Convert to neural network input
            auto input = canvasToNeuralNetInput(downscaled);
            
            // Get predictions
            auto confidences = net.forward(input);
            lastPredictionConfidences = confidences;
            lastPredictedDigit = net.predict(input);
            predictPressed = true;
        }
        
        // Handle clear button
        Rectangle clearButtonRect = {(float)clearButtonX, (float)clearButtonY, 
                                     (float)buttonWidth, (float)buttonHeight};
        bool clearHovered = CheckCollisionPointRec(GetMousePosition(), clearButtonRect);
        
        if (clearHovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            clearCanvas();
            predictPressed = false;
            lastPredictionConfidences.clear();
            lastPredictedDigit = 10;
        }
        
        // Rendering
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        // Draw title
        DrawText("Digit Recognizer", 10, -2, 16, DARKBLUE);
        
        // Draw the canvas
        renderCanvas(canvasOffsetX, canvasOffsetY);
        
        // Draw predict button
        DrawRectangleRec(predictButtonRect, predictHovered ? LIGHTGRAY : GRAY);
        DrawRectangleLinesEx(predictButtonRect, 2, BLACK);
        DrawText("Predict", predictButtonX + 15, predictButtonY + 10, 16, BLACK);
        
        // Draw clear button
        DrawRectangleRec(clearButtonRect, clearHovered ? ORANGE : RED);
        DrawRectangleLinesEx(clearButtonRect, 2, BLACK);
        DrawText("Clear", clearButtonX + 25, clearButtonY + 10, 16, WHITE);
        
        // Draw prediction results
        if (predictPressed && lastPredictedDigit < 10) {
            DrawText("Digit:", predictionX, predictionY, 12, DARKBLUE);
            
            // Draw the predicted digit in large text
            std::string digitStr = std::to_string(lastPredictedDigit);
            DrawText(digitStr.c_str(), predictionX + 60, predictionY - 5, 28, GREEN);
            
            // Draw confidence percentages
            int confidenceY = predictionY + 35;
            DrawText("Confidence:", predictionX, confidenceY, 12, DARKBLUE);
            confidenceY += 15;
            
            for (int i = 0; i < 10 && i < (int)lastPredictionConfidences.size(); i++) {
                double confidence = lastPredictionConfidences[i] * 100.0;
                
                // Create formatted string with 1 decimal place to save space
                std::stringstream ss;
                ss << std::fixed << std::setprecision(1) << confidence;
                std::string confStr = std::to_string(i) + ":" + ss.str() + "%";
                
                // Highlight the predicted digit
                Color textColor = (i == lastPredictedDigit) ? GREEN : BLACK;
                DrawText(confStr.c_str(), predictionX + 5, confidenceY, 16, textColor);
                
                confidenceY += 20;
            }
        }
        
        EndDrawing();
    }
    
    CloseWindow();
    return 0;
}