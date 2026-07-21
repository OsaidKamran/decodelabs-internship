import cv2
import numpy as np


def main():
    # 1. Define paths and network parameters
    image_path = "DSC_0002.jpg"
    prototxt_path = "MobileNetSSD_deploy.prototxt"
    model_path = "MobileNetSSD_deploy.caffemodel"
    minimum_confidence = 0.5

    # PASCAL VOC class labels that MobileNet SSD was trained to detect
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # Assign random colors to each class for the bounding boxes
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # 2. Load the model
    print("[INFO] Loading Deep Learning model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # 3. Read and pre-process the image
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Could not load image. Check the file path.")
        return

    (h, w) = image.shape[:2]
    # Convert image to a blob (Standard MobileNet input size is 300x300, scale factor 0.007843)
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 0.007843, (300, 300), 127.5)

    # 4. Pass the blob through the network to get detections
    print("[INFO] Computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    # 5. Loop over the detections and draw bounding boxes
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > minimum_confidence:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the prediction on the image
            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(image, (startX, startY),
                          (endX, endY), COLORS[idx], 2)

            # Position the label text above the bounding box
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # 6. Display the final output
    cv2.imshow("Advanced Vision Engine - Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
