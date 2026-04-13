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
const int BRUSH_SIZE = 8;

vector<vector<uint8_t>> canvas;

struct InkBox {
    int minX;
    int minY;
    int maxX;
    int maxY;
    bool valid;
};

InkBox findInkBoundingBox(const vector<vector<uint8_t>>& srcCanvas) {
    const int size = static_cast<int>(srcCanvas.size());
    InkBox box{size, size, -1, -1, false};

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (srcCanvas[y][x] > 0) {
                box.minX = min(box.minX, x);
                box.minY = min(box.minY, y);
                box.maxX = max(box.maxX, x);
                box.maxY = max(box.maxY, y);
                box.valid = true;
            }
        }
    }

    return box;
}

double sampleBilinear(const vector<vector<uint8_t>>& srcCanvas, double x, double y) {
    const int size = static_cast<int>(srcCanvas.size());
    x = clamp(x, 0.0, static_cast<double>(size - 1));
    y = clamp(y, 0.0, static_cast<double>(size - 1));

    const int x0 = static_cast<int>(floor(x));
    const int y0 = static_cast<int>(floor(y));
    const int x1 = min(x0 + 1, size - 1);
    const int y1 = min(y0 + 1, size - 1);

    const double tx = x - static_cast<double>(x0);
    const double ty = y - static_cast<double>(y0);

    const double v00 = static_cast<double>(srcCanvas[y0][x0]);
    const double v10 = static_cast<double>(srcCanvas[y0][x1]);
    const double v01 = static_cast<double>(srcCanvas[y1][x0]);
    const double v11 = static_cast<double>(srcCanvas[y1][x1]);

    const double top = v00 * (1.0 - tx) + v10 * tx;
    const double bottom = v01 * (1.0 - tx) + v11 * tx;
    return top * (1.0 - ty) + bottom * ty;
}

vector<vector<uint8_t>> alignByCenterOfMass(const vector<vector<uint8_t>>& img) {
    const int size = static_cast<int>(img.size());
    double totalMass = 0.0;
    double sumX = 0.0;
    double sumY = 0.0;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const double mass = static_cast<double>(img[y][x]);
            totalMass += mass;
            sumX += static_cast<double>(x) * mass;
            sumY += static_cast<double>(y) * mass;
        }
    }

    if (totalMass <= 0.0) {
        return img;
    }

    const double center = (static_cast<double>(size) - 1.0) * 0.5;
    const double comX = sumX / totalMass;
    const double comY = sumY / totalMass;

    const int shiftX = static_cast<int>(round(center - comX));
    const int shiftY = static_cast<int>(round(center - comY));

    vector<vector<uint8_t>> shifted(size, vector<uint8_t>(size, 0));
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const int dstX = x + shiftX;
            const int dstY = y + shiftY;
            if (dstX >= 0 && dstX < size && dstY >= 0 && dstY < size) {
                shifted[dstY][dstX] = img[y][x];
            }
        }
    }

    return shifted;
}

// Function to downscale 280x280 to 28x28
vector<vector<uint8_t>> downscaleCanvas(const vector<vector<uint8_t>>& srcCanvas) {
    const int dstSize = 28;
    const int targetDigitSize = 20;
    vector<vector<uint8_t>> result(dstSize, vector<uint8_t>(dstSize, 0));

    const InkBox box = findInkBoundingBox(srcCanvas);
    if (!box.valid) {
        return result;
    }

    const int srcWidth = box.maxX - box.minX + 1;
    const int srcHeight = box.maxY - box.minY + 1;
    const double scale = static_cast<double>(targetDigitSize) /
                         static_cast<double>(max(srcWidth, srcHeight));

    const int scaledWidth = max(1, static_cast<int>(round(srcWidth * scale)));
    const int scaledHeight = max(1, static_cast<int>(round(srcHeight * scale)));

    const int offsetX = (dstSize - scaledWidth) / 2;
    const int offsetY = (dstSize - scaledHeight) / 2;

    for (int y = 0; y < scaledHeight; ++y) {
        for (int x = 0; x < scaledWidth; ++x) {
            const double srcX = static_cast<double>(box.minX) +
                                (static_cast<double>(x) + 0.5) *
                                (static_cast<double>(srcWidth) / static_cast<double>(scaledWidth));
            const double srcY = static_cast<double>(box.minY) +
                                (static_cast<double>(y) + 0.5) *
                                (static_cast<double>(srcHeight) / static_cast<double>(scaledHeight));

            const double sampled = sampleBilinear(srcCanvas, srcX, srcY);
            result[offsetY + y][offsetX + x] =
                static_cast<uint8_t>(clamp(sampled, 0.0, 255.0));
        }
    }

    return alignByCenterOfMass(result);
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

void drawStrokeSegment(int x0, int y0, int x1, int y1) {
    const int steps = max(abs(x1 - x0), abs(y1 - y0));
    if (steps == 0) {
        drawBrushStroke(x0, y0);
        return;
    }

    for (int i = 0; i <= steps; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(steps);
        const int x = static_cast<int>(round(static_cast<double>(x0) + t * static_cast<double>(x1 - x0)));
        const int y = static_cast<int>(round(static_cast<double>(y0) + t * static_cast<double>(y1 - y0)));
        drawBrushStroke(x, y);
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

bool canvasHasInk() {
    for (int y = 0; y < CANVAS_SIZE; y++) {
        for (int x = 0; x < CANVAS_SIZE; x++) {
            if (canvas[y][x] > 0) {
                return true;
            }
        }
    }
    return false;
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
    bool showEmptyCanvasHint = false;
    vector<double> lastPredictionConfidences;
    uint8_t lastPredictedDigit = 10;  // Invalid initially
    bool wasDrawing = false;
    int lastDrawX = 0;
    int lastDrawY = 0;
    
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
                if (wasDrawing) {
                    drawStrokeSegment(lastDrawX, lastDrawY, relativeX, relativeY);
                } else {
                    drawBrushStroke(relativeX, relativeY);
                }
                lastDrawX = relativeX;
                lastDrawY = relativeY;
                wasDrawing = true;
                showEmptyCanvasHint = false;
            } else {
                wasDrawing = false;
            }
        } else {
            wasDrawing = false;
        }
        
        Rectangle predictButtonRect = {(float)predictButtonX, (float)predictButtonY, 
                                       (float)buttonWidth, (float)buttonHeight};
        bool predictHovered = CheckCollisionPointRec(GetMousePosition(), predictButtonRect);
        
        if (predictHovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            if (!canvasHasInk()) {
                showEmptyCanvasHint = true;
                predictPressed = false;
                lastPredictionConfidences.clear();
                lastPredictedDigit = 10;
            } else {
                auto downscaled = downscaleCanvas(canvas);
                auto input = canvasToNeuralNetInput(downscaled);
                auto confidences = net.forward(input);
                lastPredictionConfidences = confidences;
                lastPredictedDigit = net.predict(input);
                predictPressed = true;
                showEmptyCanvasHint = false;
            }
        }
        
        // Handle clear button
        Rectangle clearButtonRect = {(float)clearButtonX, (float)clearButtonY, 
        (float)buttonWidth, (float)buttonHeight};
        bool clearHovered = CheckCollisionPointRec(GetMousePosition(), clearButtonRect);
        
        if (clearHovered && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            clearCanvas();
            predictPressed = false;
            showEmptyCanvasHint = false;
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

        if (showEmptyCanvasHint) {
            DrawText("Draw a digit first", predictButtonX - 10, clearButtonY + 56, 16, MAROON);
        }
        
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