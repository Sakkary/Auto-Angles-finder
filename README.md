# Auto-Angles-finder

Measure angles between pipe connections around a circular node in engineering
drawings.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install opencv-python numpy flask
```

## Run

```bash
python angle_finder.py path/to/image.png
```

## Web app

```bash
export FLASK_APP=app.py
flask run
```

Open the URL printed by Flask (typically http://127.0.0.1:5000) and upload an
image to see the detected angles and annotated output.

## Quick check

```bash
python -m py_compile angle_finder.py
```
