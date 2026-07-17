"""
vision_engine.py

DecodeLabs Project 4: Static Object Detection Pipeline
Features: Auto-model acquisition, Static image ingestion, 
MobileNet-SSD inference, and an adjusted 15% Confidence testing gate.
"""

import cv2
import numpy as np
import urllib.request
import os

# ==========================================
# SETUP: Model Acquisition & Definitions
# ==========================================
# Active repository URLs for MobileNet-SSD
PROTOTXT_URL = "https://raw.githubusercontent.com/TheNsBhasin/DNN_Object_Detection/master/MobileNetSSD_deploy.prototxt.txt"
MODEL_URL = "https://raw.githubusercontent.com/TheNsBhasin/DNN_Object_Detection/master/MobileNetSSD_deploy.caffemodel"

PROTOTXT_PATH = "MobileNetSSD_deploy.prototxt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"

# Target Image
IMAGE_PATH = "DSC_0002.jpg"

# MobileNet-SSD recognizes 20 specific objects (plus background)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Assign random bounding box colors for each class for visual clarity
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def download_model_files():
    """Ensures the neural network weights and architecture files exist locally."""
    if not os.path.exists(PROTOTXT_PATH):
        print("[SYSTEM] Downloading Prototxt architecture...")
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
    if not os.path.exists(MODEL_PATH):
        print("[SYSTEM] Downloading Caffe model weights (this may take a moment)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[SYSTEM] Neural Network assets verified.")

# ==========================================
# MAIN PIPELINE
# ==========================================


def main():
    print("--- INITIALIZING STATIC VISION ENGINE ---")
    download_model_files()

    # Load our serialized model from disk
    print("[SYSTEM] Loading model into cv2.dnn engine...")
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

    # STEP 1: INGESTION (Load Image)
    print(f"[SYSTEM] Ingesting image: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print(
            f"[ERROR] Could not load image at {IMAGE_PATH}. Please check the filename.")
        return

    # Resize for display purposes if the image is massive
    image = cv2.resize(
        image, (800, int(800 * image.shape[0] / image.shape[1])))
    (h, w) = image.shape[:2]

    # STEP 2: PRE-PROCESSING (Blob Construction)
    print("[SYSTEM] Constructing 4D Blob for inference...")
    # Convert the image matrix into a 4D blob, perform mean subtraction, and scale to 300x300
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # STEP 3: INFERENCE
    print("[SYSTEM] Executing neural network forward pass...")
    net.setInput(blob)
    detections = net.forward()

    # STEP 4: FILTERING & OUTPUT (The 15% Gate)
    found_objects = False
    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Testing threshold: lowered to 15% to expose low-confidence detections
        if confidence >= 0.15:
            found_objects = True

            # Extract the index of the class label
            idx = int(detections[0, 0, i, 1])

            # Calculate the bounding box (X, Y) coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Format the label output
            label = f"{CLASSES[idx]}: {confidence * 100:.1f}%"
            print(f"[MATCH] Detected: {label}")

            # Draw the bounding box and label on the image
            cv2.rectangle(image, (startX, startY),
                          (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    if not found_objects:
        print("[SYSTEM] No objects met the 15% confidence threshold.")

    # Show the static output
    print("[SYSTEM] Displaying output window. Press any key to close.")
    cv2.imshow("DecodeLabs Static Output", image)

    # Wait indefinitely until the user presses a key, then close
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
