
import os
from dbr import BarcodeReader, EnumBarcodeFormat, EnumBarcodeFormat_2, BarcodeReaderError
import cv2
import re
import numpy as np
from shapely.geometry import Polygon

# Initialize the Barcode Reader with your license
reader = BarcodeReader()
reader.init_license("t0068lQAAAAxutKzSzbjUW+sdIyi5cV/NtqCrF5Klc1KOt0p4qglFiWO5/tVU8bYQ19C1DT2Q4DgL5YtszZQ22FAbrKw3E6Q=;t0069lQAAAGENyQId75HoF2W4tJRwcWB4zbH5MvHm6vWfTcQmeVbYqpL4CijH0dvz67SmmdCQuwOhERdOXrKhvMnFoBvEZvFy")  # Replace "YOUR_LICENSE_KEY" with your actual license key.

# Define the input and output paths
input_file = "pharma3.jpg"  # Path to your input image
output_dir = "SPAN"             # Directory to save output image
output_file="output.jpg"


def barcodeReader(input_file, output_file):
    """
    Reads barcodes from the input image, highlights them, and saves the output image.

    Args:
        input_file (str): Path to the input image file.
        output_file (str): Path to save the output image.

    Returns:
        list: A list of dictionaries with barcode details.
    """
    # Load the input image
    img = cv2.imread(input_file)
    if img is None:
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    # Store detected barcode details
    detected_barcodes = []

    # Loop for two passes to detect all barcode formats
    for count in range(2):
        # Configure the settings for the barcode reader
        settings = reader.get_runtime_settings()
        if count == 0:
            settings.barcode_format_ids = EnumBarcodeFormat.BF_ALL
        else:
            settings.barcode_format_ids_2 = EnumBarcodeFormat_2.BF2_ALL
        reader.update_runtime_settings(settings)

        # Decode barcodes in the image
        try:
            text_results = reader.decode_file(input_file)
            if text_results:
                for text_result in text_results:
                    if text_result.barcode_format_string:
                        # Extract barcode details
                        boxes = text_result.localization_result.localization_points
                        expanded_vertices = expand_polygon(boxes, offset=10)
                        pts = np.array(expanded_vertices, np.int32).reshape((-1, 1, 2))

                        # Draw the polygon on the image
                        cv2.polylines(img, [pts], True, (0, 0, 255), thickness=3)

                        # Store details in the result
                        detected_barcodes.append({
                            "DetailValue": re.sub(r'\W+', '', text_result.barcode_text),
                            "DetailColor": "blue",
                            "DetailName": text_result.barcode_format_string,
                            "Boxes": boxes
                        })

        except BarcodeReaderError as bre:
            print(f"Error during barcode decoding: {bre}")

    # Save the annotated image
    cv2.imwrite(output_file, img)
    print(f"Output image saved to: {output_file}")

    return detected_barcodes


def expand_polygon(vertices, offset):
    """
    Expands a polygon outward using Shapely's buffer method.

    Args:
        vertices (list): List of vertex points [(x1, y1), (x2, y2), ...].
        offset (int): Distance to expand the polygon.

    Returns:
        list: Expanded polygon vertices.
    """
    polygon = Polygon(vertices)
    expanded = polygon.buffer(offset, join_style=2)  # Join style ensures sharp corners
    return [(int(x), int(y)) for x, y in expanded.exterior.coords]


# Call the function with the provided input and output file paths
if __name__ == "__main__":
    try:
        results = barcodeReader(input_file, output_file)
        print("Detected Barcodes:")
        for result in results:
            print(result)
    except Exception as e:
        print(f"Error: {e}")
