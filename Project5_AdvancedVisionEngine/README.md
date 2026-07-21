# Project 5: Advanced Vision Engine (Deep Learning Inference System)
An advanced computer vision architecture leveraging OpenCV's Deep Neural Network (DNN) module and a MobileNet-SSD to perform multi-class object localization and classification on unstructured visual data.

## Features
- **DNN Blob Pre-Processing:** Employs advanced spatial scaling and mean subtraction to format raw visual data into multi-dimensional blobs optimized for network ingestion.
- **Multi-Class Localization:** Dynamically identifies and bounds up to 20 distinct object categories utilizing PASCAL VOC trained weights.
- **Dynamic Confidence Annotation:** Generates color-coded bounding geometries overlaid with exact class predictions and percentage-based confidence scores.
- **Optimized Forward Pass:** Executes an efficient forward pass through the Caffe-based Single Shot Detector to ensure rapid and accurate inference.

## Setup
1. `python -m venv venv`
2. `source venv/bin/activate` (or `.\venv\Scripts\activate`)
3. `python -m pip install opencv-python==4.10.0.84 numpy`
4. `python vision_engine.py`

*Engineered by Syed Osaid Kamran*