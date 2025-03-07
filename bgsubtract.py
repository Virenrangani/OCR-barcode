

import numpy as np
import cv2
import json

def preprocess_image(img):
    """Enhanced preprocessing to isolate objects."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Gradient-based edge enhancement (Canny or Sobel)
    edges = cv2.Canny(gray, 50, 150)  # Adjust thresholds (50, 150)

    # 2. Morphological operations (more targeted)
    kernel = np.ones((5, 5), np.uint8)  # Smaller kernel
    dilated = cv2.dilate(edges, kernel, iterations=1)  # Adjust iterations
    morph = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)  # Adjust iterations

    # 3. Further noise removal with Gaussian blur (optional, but can help)
    blurred = cv2.GaussianBlur(morph, (5, 5), 0)  # Adjust kernel size if needed

    # 4. Thresholding (using the result of edge detection and morphological operations)
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return blurred   # Return the thresholded image


def detect_and_draw_contours(img):
    """Detects and draws contours, identifies squares, and saves data."""
    processed = preprocess_image(img)  # Use the improved preprocessing

    # Find contours (using RETR_EXTERNAL for outer contours)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 1000  # Increased minimum area (adjust as needed)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    contour_image = img.copy()
    all_coordinates = []

    for cnt in filtered_contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)  # Adjust epsilon
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:  # Potential square
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.7 <= aspect_ratio <= 1.3:  # Wider aspect ratio range
                cv2.drawContours(contour_image, [approx], -1, (0, 0, 255), 3)  # Red for squares
                contour_points = [point[0].tolist() for point in approx]
                all_coordinates.append({"type": "square", "coordinates": contour_points})
            else:  # Quadrilateral but not square
                cv2.drawContours(contour_image, [approx], -1, (0, 255, 0), 2)  # Green
                contour_points = [point[0].tolist() for point in approx]
                all_coordinates.append({"type": "quadrilateral", "coordinates": contour_points})

        else:  # Other contour
            cv2.drawContours(contour_image, [approx], -1, (0, 255, 0), 2)  # Green
            contour_points = [point[0].tolist() for point in approx]
            all_coordinates.append({"type": "other", "coordinates": contour_points})

    cv2.imwrite("output2.6.jpg", contour_image)
    with open("contours_coordinates.json", "w") as f:
        json.dump({"contours": all_coordinates}, f, indent=4)
    print("Contours detected and saved.")


# Load image and process
img = cv2.imread("o2.jpg")  # Replace with your image path
detect_and_draw_contours(img)