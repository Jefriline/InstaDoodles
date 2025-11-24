# InstaDoodles

Automated drawing tool that converts any image into live doodle art drawn by your mouse cursor.

## What It Does

InstaDoodles automatically draws an image on your screen (like in MS Paint) using Python and computer vision. It transforms pictures into contour art with professional quality, extreme speed, and superior artistic results.

## Features

- **Fill Mode**: Draws the complete fill of the image using optimized scanline algorithms
- **Literal Photocopy Mode**: Exact pixel-perfect copy for line art drawings (Detail >= 9)
- **Artistic Mode**: Generates artistic strokes with textures and shadows (Detail < 9)
- **Extreme Speed**: Uses direct Windows API injection for ultra-fast drawing
- **True Transparency**: Completely transparent window, only shows the black parts of the image
- **File Selection**: Load any image from your file system
- **Real-time Rendering**: Progressive drawing updates while executing
- **Modern UI**: CustomTkinter interface with dark mode and intuitive controls

## Tech Stack

- Python 3.10+
- OpenCV - Multi-scale edge detection and image processing
- CustomTkinter - Modern UI with dark mode
- Numba - JIT optimization for 50x-100x faster processing
- NumPy - Efficient image handling
- SciPy - Advanced artistic filling algorithms
- ctypes/win32api - Direct Windows input injection for extreme speed
- PIL/Pillow - Image processing with transparency

## Installation

### Clone the repository

```bash
git clone https://github.com/Jefriline/InstaDoodles.git
cd InstaDoodles
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python doddles.py
```

## Usage

1. **Select Image**: Click "Seleccionar Imagen" to load an image from your computer
2. **Adjust Size**: Use the "Tamaño (%)" slider to scale the image
3. **Adjust Detail**: 
   - **Detail 9-10**: "Literal Photocopy" mode - Exact pixel-perfect copy (slower but perfect)
   - **Detail 1-8**: "Artistic" mode - Generates artistic strokes with textures (faster)
4. **Adjust Transparency**: Use the "Transparencia (%)" slider to better see the image
5. **Prepare Drawing**: Click "Preparar Dibujo" to process the image
6. **Start Drawing**: Click "Comenzar Dibujo" and align the window where you want to draw
7. **Wait**: The program will automatically draw the image. Press Ctrl+C or the "Cancelar" button to stop

## Drawing Modes

### Fill Mode (Default)
Draws the complete fill of the image using optimized scanline algorithms with memory tracking to avoid redrawing already completed areas.

### Literal Photocopy Mode (Detail >= 9)
For clean line art drawings (like a drawn pineapple):
- Clean binarization without inventing lines
- No dithering or invented textures
- Spacing = 1 for maximum resolution
- Exact pixel-perfect copy

### Artistic Mode (Detail < 9)
For complex images (like a spider with fine hairs):
- Sharpening to highlight fine details
- Floyd-Steinberg dithering to preserve textures
- Artistic strokes with weight variation

## Optimizations

- **Numba JIT**: Critical functions compiled to machine code for extreme speed
- **ctypes/Windows API**: Direct input injection, no PyAutoGUI overhead
- **Memory Mapping**: Tracking system to avoid redrawing already completed areas
- **Two-Pass Filling**: Main pass with normal spacing + fine pass (1px) to cover small gaps

## Requirements

- Windows 10/11
- Python 3.10, 3.11 or 3.12
- All dependencies listed in `requirements.txt`

## Notes

- The application works best with high contrast images
- For line art, use Detail 9-10 for best fidelity
- For complex images, use Detail 5-8 for balanced speed and quality
- The program will automatically close when the drawing is complete

## Repository

https://github.com/Jefriline/InstaDoodles.git

## Credits

- Original fork: [manitdangal/InstaDoodles](https://github.com/manitdangal/InstaDoodles)
- Improvements and optimizations: Jefriline
