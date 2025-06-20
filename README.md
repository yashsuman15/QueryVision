# QueryVision

An object detection application using OWL-ViT and OWLv2 models for text-based and image-based object detection.

## What it does

QueryVision detects objects in images using two methods:

1. **Text-based detection** - Find objects by describing them with text (e.g., "coin", "person", "car")
2. **Image-based detection** - Find objects by providing a reference image of what you're looking for

## Requirements

- Python 3.10+
- Dependencies listed in `pyproject.toml`

## Installation

Before installation create a conda env using 

```bash
conda create --name queryvision python=3.10
```
then

```bash
conda activate queryvision
```

### Step-1: using pip
```bash
pip install -e .
```

```bash
python main.py
```

### Step-2: using uv

```bash
uv sync
```

```bash
uv run main.py
```

## Usage

1. Choose detection method:
   - Option 1: Text-based detection
   - Option 2: Image-based detection

2. Select model:
   - OWLv2 (recommended)
   - OWL-ViT

3. For text-based detection:
   - Select image file
   - Results show detected objects with bounding boxes

4. For image-based detection:
   - Select reference image
   - Crop the target object using interactive tool
   - Select target image to search in
   - Results show similar objects found

## Configuration

Edit `config/constants.py` to adjust:
- Detection thresholds
- Device settings (CPU/GPU)

## Output

- Console output with detection results
- Annotated images saved as JPG files
- Visual display of detections with bounding boxes