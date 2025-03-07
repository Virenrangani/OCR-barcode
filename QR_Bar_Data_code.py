import cv2
import numpy as np
from pyzbar.pyzbar import decode

# General method to decode QR, Barcode, and Data Matrix codes
def scan_codes(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Image not found")
        return
    
    # Decode all types of codes (QR, Barcode, Data Matrix)
    decoded_objects = decode(img)
    
    if not decoded_objects:
        print("No barcode or QR/Data Matrix code detected")
    else:
        for obj in decoded_objects:
            # Get the type of the detected code (QR, Barcode, or Data Matrix)
            code_type = obj.type
            code_data = obj.data.decode('utf-8')
            
            # Print the code type and data
            print(f"Detected {code_type}: {code_data}")
            
            # Draw rectangle around the detected code
            rect_points = obj.polygon
            if len(rect_points) == 4:
                pts = [tuple(point) for point in rect_points]
                cv2.polylines(img, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
            else:
                # If not a 4-point polygon (like a barcode), use the bounding box
                x, y, w, h = obj.rect
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the image with highlighted detected codes
    cv2.imshow("Code Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Take the image path as input or use a fixed path
    image_path = 'your_path.jpg'  # Replace with the actual image file path
    scan_codes(image_path)