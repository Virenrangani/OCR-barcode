
import cv2
import numpy as np
from paddleocr import PaddleOCR

def preprocess_image(image_path):
    
    image = cv2.imread(image_path)
   
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    
    resized = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Adjust brightness and contrast slightly
    alpha = 1.2  # Contrast control (1.0-2.0)
    beta = 30    # Brightness control (0-50)
    bright_contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Apply slight Gaussian blur to reduce noise without affecting clarity
    blurred = cv2.GaussianBlur(bright_contrast, (3, 3), 0)

    # Use adaptive thresholding to binarize the image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 5
    )

    # Apply a very light morphological operation to smooth text edges
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph_kernel)

    return resized, morphed

def perform_ocr(image, ocr_model):
    # Save the preprocessed image for OCR input
    temp_image_path = "processed_image_7.jpg"
    cv2.imwrite(temp_image_path, image)

    # Perform OCR using PaddleOCR
    result = ocr_model.ocr(temp_image_path, cls=True)

    # Extract and print detected text
    print("Detected Text:")
    for line in result[0]:
        print(f"Text: {line[1][0]} | Confidence: {line[1][1]}")

    return result

def main():
    try:
        # Image path
        image_path = 'e7.jpg'  # Update with your file path

        # Step 1: Preprocess the image
        resized_image, processed_image = preprocess_image(image_path)

        # Step 2: Initialize PaddleOCR
        ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

        # Step 3: Perform OCR
        ocr_results = perform_ocr(processed_image, ocr_model)

        # Display the results visually
        cv2.imshow("Resized Image", resized_image)
        cv2.imshow("Processed Image", processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    main()

