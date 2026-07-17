# Project 4: Vision Engine (Object Detection Pipeline)
An industrial-grade object detection pipeline utilizing `OpenCV` and `MobileNet-SSD` to ingest and process unstructured visual data.

## Features
- **Auto-Model Acquisition:** Automatically verifies and downloads necessary neural network weights and architecture files.
- **Pre-Processing:** Implements Grayscale conversion and Blob construction for neural network compatibility.
- **Inference Pipeline:** Utilizes a Single Shot Detector (SSD) architecture for real-time object classification.
- **Industrial Filtering:** Implements an 80% confidence threshold gate to eliminate false positives and prevent hallucinations.

## Setup
1. `python -m venv venv`
2. `source venv/bin/activate` (or `.\venv\Scripts\activate`)
3. `pip install opencv-python==4.10.0.84 numpy`
4. `python vision_engine.py`

*Engineered by Syed Osaid Kamran*