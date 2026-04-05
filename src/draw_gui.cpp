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

using namespace std;

const int CANVAS_SIZE = 280;
const int DISPLAY_SCALE = 1;  // Display the canvas at 1x scale to fit in 600x400
const int DISPLAY_WIDTH = CANVAS_SIZE * DISPLAY_SCALE;
const int DISPLAY_HEIGHT = CANVAS_SIZE * DISPLAY_SCALE;
const int BRUSH_SIZE = 10;

vector<vector<uint8_t>> canvas;

// Function to downscale 280x280 to 28x28
vector<vector<uint8_t>> downscaleCanvas(const vector<vector<uint8_t>>& srcCanvas) {
    const int srcSize = 280;
    const int dstSize = 28;
    const int scale = srcSize / dstSize;  // Should be 10
    
    vector<vector<uint8_t>> result(dstSize, vector<uint8_t>(dstSize, 0));
    
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
vector<double> canvasToNeuralNetInput(const vector<vector<uint8_t>>& downscaledCanvas) {
    vector<double> input;
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
    canvas = vector<vector<uint8_t>>(CANVAS_SIZE, vector<uint8_t>(CANVAS_SIZE, 0));
    
    NeuralNetwork net(784, 128, 10, 0.1);
    if (!net.loadModel("models/mnist_model.bin")) {
        cerr << "Failed to load model!" << endl;
        return 1;
    }
    
    const int screenWidth = 600;
    const int screenHeight = 400;
    InitWindow(screenWidth, screenHeight, "Digit Recognizer - Draw to Predict");
    SetTargetFPS(60);
    
    bool predictPressed = false;
    vector<double> lastPredictionConfidences;
    uint8_t lastPredictedDigit = 10;  // Invalid initially
    
    const int canvasOffsetX = 10;
    const int canvasOffsetY = 52;
    const int buttonWidth = 100;
    const int buttonHeight = 40;
    const int predictButtonX = canvasOffsetX + DISPLAY_WIDTH + 15;
    const int predictButtonY = canvasOffsetY;
    const int clearButtonX = predictButtonX;
    const int clearButtonY = predictButtonY + buttonHeight + 10;
    
    const int predictionX = predictButtonX;
    const int predictionY = clearButtonY + buttonHeight + 10;
    const int confidenceX = predictionX + 125;
    const int confidenceYStart = predictButtonY;
    
    while (!WindowShouldClose()) {
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            int mouseX = GetMouseX();
            int mouseY = GetMouseY();
            
            if (mouseX >= canvasOffsetX && mouseX < canvasOffsetX + DISPLAY_WIDTH &&
                mouseY >= canvasOffsetY && mouseY < canvasOffsetY + DISPLAY_HEIGHT) {
                
                int relativeX = mouseX - canvasOffsetX;
                int relativeY = mouseY - canvasOffsetY;
                drawBrushStroke(relativeX, relativeY);
            }
        }
        
        Rectangle predictButtonRect = {(float)predictButtonX, (float)predictButtonY, 
                                       (float)buttonWidth, (float)buttonHeight};
        bool predictHovered = CheckCollisionPointRec(GetMousePosition(), predictButtonRect);
        
        if (predictHovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            auto downscaled = downscaleCanvas(canvas);
            
            auto input = canvasToNeuralNetInput(downscaled);
            
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
        
        BeginDrawing();
        ClearBackground(RAYWHITE);
        
        DrawText("Digit Recognizer", 10, 10, 26, DARKBLUE);
        DrawText("Draw a digit, then click Predict", 10, 36, 15, DARKGRAY);
        DrawLine(8, 48, screenWidth - 8, 48, LIGHTGRAY);
        
        // Draw canvas
        renderCanvas(canvasOffsetX, canvasOffsetY);
        
        DrawRectangleRec(predictButtonRect, predictHovered ? LIGHTGRAY : GRAY);
        DrawRectangleLinesEx(predictButtonRect, 2, BLACK);
        DrawText("Predict", predictButtonX + 13, predictButtonY + 9, 18, BLACK);
        
        // Draw clear button
        DrawRectangleRec(clearButtonRect, clearHovered ? ORANGE : RED);
        DrawRectangleLinesEx(clearButtonRect, 2, BLACK);
        DrawText("Clear", clearButtonX + 23, clearButtonY + 9, 18, WHITE);
        
        // Draw prediction results
        if (predictPressed && lastPredictedDigit < 10) {
            DrawText("Digit:", predictionX, predictionY, 17, DARKBLUE);
            
            // Draw the predicted digit in large text
            string digitStr = to_string(lastPredictedDigit);
            DrawText(digitStr.c_str(), predictionX + 60, predictionY - 5, 28, GREEN);
            
            // Draw confidence percentages in a separate right-side column.
            int confidenceY = confidenceYStart;
            DrawText("Confidence:", confidenceX, confidenceY, 17, DARKBLUE);
            confidenceY += 18;
            
            for (int i = 0; i < 10 && i < (int)lastPredictionConfidences.size(); i++) {
                double confidence = lastPredictionConfidences[i] * 100.0;
                
                // Create formatted string with 1 decimal place to save space
                stringstream ss;
                ss << fixed << setprecision(1) << confidence;
                string spacing = (i == 1) ? "    " : "   ";
                string confStr = to_string(i) + ":" + spacing + ss.str() + "%";
                
                Color textColor = (i == lastPredictedDigit) ? GREEN : BLACK;
                DrawText(confStr.c_str(), confidenceX, confidenceY, 18, textColor);
                
                confidenceY += 20;
            }
        }
        
        EndDrawing();
    }
    
    CloseWindow();
    return 0;
}