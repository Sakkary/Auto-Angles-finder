# Auto Angles Finder

Automatically detect a circular node in engineering drawings and measure the angles between connected pipe directions. The tool identifies the circle center, determines connection directions, and outputs both numeric results and an annotated image for verification.

This project is intended for learning computer vision, geometry, and practical engineering automation.

---

## Features

- Detects the main circular node
- Detects connections that meet the circle
- Computes:
  - absolute angle of each connection
  - sector angles between consecutive connections
- Sum of sector angles is approximately 360 degrees
- Produces an annotated image with circle and angles drawn
- Can be used:
  - from the command line
  - as a small web application using Flask

---

## Requirements

- Python 3.9 or newer
- pip package manager
- Internet connection only required when installing dependencies

On some systems OpenCV may require additional system libraries such as `libgl1`. If installation fails, check your operating system documentation.

---

## Installation

Create and activate a virtual environment.

```bash
python -m venv .venv

# Linux or macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies.

```bash
pip install opencv-python numpy flask
```

Alternatively, if a `requirements.txt` file exists.

```bash
pip install -r requirements.txt
```

---

## Run from the command line

Run the angle detection on a single image.

```bash
python angle_finder.py path/to/image.png
```

The program prints:

- detected circle center
- detected radius
- connection angles in degrees
- sector angles in degrees
- sum of sector angles

It also writes an annotated image to the working directory:

```
annotated_output.png
```

Open it to visually verify that the geometry matches expectations.

---

## Web application mode

You can run a simple website locally to upload images through the browser.

Start Flask:

```bash
export FLASK_APP=app.py        # Windows: set FLASK_APP=app.py
flask run
```

Then open the printed URL in your browser, usually:

```
http://127.0.0.1:5000
```

Upload an image. The page will show:

- JSON results
- annotated image with detected angles

---

## How it works

High level processing pipeline:

1. Load the image
2. Convert to grayscale
3. Apply blur and detect edges
4. Detect the most likely circle using Hough Circle Transform
5. Detect line segments using probabilistic Hough Lines
6. Select segments that connect to the detected circle
7. Compute direction angles relative to the circle center
8. Sort angles and compute sector angles that form a full rotation

The method does not rely on a fixed color. Detection is based on geometry.

---

## Limitations

Current limitations include:

- noisy drawings may confuse circle detection
- very thick lines may merge during edge detection
- images with several similar circles may require tuning
- low resolution scans reduce accuracy
- output is angular only, not length or scale aware

You can tune parameters inside `angle_finder.py`, for example:

- minimum and maximum circle radius
- Canny thresholds
- Hough transform parameters
- connection tolerance ratio
- angle merge tolerance

---

## Contributing

Contributions are welcome, including:

- parameter tuning
- performance improvements
- support for multiple circles
- better visualization
- improved web interface
- documentation updates
- additional example images

---

## License

This project is provided under the license included in this repository. Review the license terms before using the software in a commercial context.

---

## Acknowledgements

Built using:

- OpenCV
- NumPy
- Flask

and classical ideas from computer vision and geometry.
