# Hairline Tracker

A GUI application for tracking and analyzing hairline changes over time using computer vision.

Check out the demo video here:

[![thumbnail](https://img.youtube.com/vi/JtgaoU93Doo/0.jpg)](https://youtu.be/JtgaoU93Doo)

## Features

- Upload and process images to track hairline position
- View individual processed images with hairline detection
- Compare any two images side-by-side
- Analyze changes with metrics and visualizations
- Track trends over time with progress charts

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Pillow
- tkinter
- matplotlib (for visualization)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/akshatmardia/hairline-tracker.git
   cd hairline-tracker
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv ./path-to-your-folder
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```

2. Using the application:
   - Click "Add New Image" to select and process a photo
   - Select images from the list to view details
   - Use "Compare Images" to see side-by-side comparisons
   - Click "Analyze Progress" to see charts and metrics of changes over time

## How It Works

The application uses computer vision techniques to:
1. Detect facial landmarks
2. Measure hairline position
3. Track changes between images
4. Calculate metrics like recession/advancement percentage

## Data Storage

All processed images and analysis data are stored locally in the `hairline_data` directory.

## Notes

- For best results,
   - Use well-lit (try using flash) photos taken from the same angle
   - Use photos with your entire face to include all facial features
   - Use a comb to push your hair back (wet hair make it easier to eliminate stray strands)
   - Use a plain white background
- Consistent lighting and positioning will improve tracking accuracy
- At least two images are required for comparison and analysis features
