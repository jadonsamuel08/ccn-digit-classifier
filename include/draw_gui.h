#ifndef DRAW_GUI_H
#define DRAW_GUI_H

#include <vector>
#include <cstdint>

// Canvas dimensions
const int CANVAS_SIZE = 280;
const int NN_INPUT_SIZE = 28;
const int BRUSH_SIZE = 4;

// Function declarations
std::vector<std::vector<uint8_t>> downscaleCanvas(const std::vector<std::vector<uint8_t>>& srcCanvas);
std::vector<double> canvasToNeuralNetInput(const std::vector<std::vector<uint8_t>>& downscaledCanvas);
void drawBrushStroke(int centerX, int centerY);
void renderCanvas(int offsetX, int offsetY);
void clearCanvas();

#endif
