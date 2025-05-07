import cv2
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt
from fast_plate_ocr import ONNXPlateRecognizer


YOLO_MODEL_PATH = "best (2).pt"  
IMAGE_PATH = "test images/021d0fd9b41f8d4b_jpg.rf.556b29b126d944cfbbc7ada5a412a710.jpg" # Image to process
OCR_MODEL_NAME = "global-plates-mobile-vit-v2-model" #ocr model version
APPLY_PREPROCESSING = True # Allows to easily check effect of preprocessing 


# Load fine-tuned YOLO model
model = YOLO(YOLO_MODEL_PATH)

# Load image
image = cv2.imread(IMAGE_PATH)

# Load OCR model
ocr_model = ONNXPlateRecognizer(OCR_MODEL_NAME)


if ocr_model:
    # Run YOLO detection
    print("Running YOLO detection...")
    results = model(IMAGE_PATH)
    if not results or not hasattr(results[0], 'boxes'):
        print("Error: YOLO detection did not return expected results.")
        boxes = []
    else:
        boxes = results[0].boxes.xyxy.cpu().numpy()
    print(f"Detected {len(boxes)} potential plates.")

    # Copy of image for visualization
    vis_img = image.copy()
    h, w, _ = image.shape

    # Loop through detected plates
    print("Processing detected boxes...")
    detection_count = 0
    for i, box in enumerate(boxes):
        detection_count += 1
        x1, y1, x2, y2 = map(int, box)
        print(f"Box {i}: ({x1}, {y1}, {x2}, {y2})")

        # Add some padding to make sure that necessary content of license plate is covered
        pad = 5
        x1p = max(x1 - pad, 0)
        y1p = max(y1 - pad, 0)
        x2p = min(x2 + pad, w)
        y2p = min(y2 + pad, h)

        # Crop license plate
        crop = image[y1p:y2p, x1p:x2p]

        if crop.size == 0:
            print(f"  Warning: Empty crop for box {i}. Skipping.")
            continue

        # Convert to grayscale
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        print(f"  Converted crop to grayscale, shape: {gray_crop.shape}")

        # Further Preprocessing
        if APPLY_PREPROCESSING:
            print("  Applying CLAHE preprocessing...")
            # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4)) # clipLimit and tilegridsize could be adjusted 
            processed_gray_crop = clahe.apply(gray_crop)
            ocr_input = processed_gray_crop # Use preprocessed image for OCR
            # Visualize the preprocessed crop
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(gray_crop, cmap='gray')
            plt.title(f'Grayscale Box {i}')
            plt.subplot(1, 2, 2)
            plt.imshow(ocr_input, cmap='gray')
            plt.title(f'CLAHE Processed Box {i}')
            plt.show(block=False)
        else:
            print("  Skipping preprocessing...")
            ocr_input = gray_crop # Use original grayscale image for OCR


        # Running OCR
        plate_text = "OCR Error"
        try:
            print(f"  Running fast_plate_ocr...")
            results_ocr = ocr_model.run(ocr_input) 
            print(f"  Raw OCR Result: {results_ocr}")

            if results_ocr:
                 if isinstance(results_ocr[0], (list, tuple)) and len(results_ocr[0]) > 0:
                     plate_text = results_ocr[0][0]
                 else:
                     plate_text = str(results_ocr[0])
                 plate_text = plate_text.replace(" ", "") # Clean spaces
            else:
                 plate_text = "N/A"

        except Exception as e:
            print(f"  Error during fast_plate_ocr for box {i}: {e}")

        print(f"  Selected Text: {plate_text}")

        # Draw box and text on image
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, plate_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show Final Result
    if detection_count > 0:
        print("Displaying final image...")
        vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_rgb)
        title = "YOLO + fast-plate-ocr"
        if APPLY_PREPROCESSING:
            title += " (with CLAHE Preprocessing)"
        plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        print("No license plates were detected by YOLO.")
