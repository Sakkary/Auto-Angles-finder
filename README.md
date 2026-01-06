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
pip install opencv-python numpy
```

## Run

```bash
python angle_finder.py path/to/image.png
```

## Quick check

```bash
python -m py_compile angle_finder.py
```
