# from dbr import *
# import cv2
# from PIL import Image, ImageDraw
# import re
# import numpy as np
# from shapely.geometry import Polygon
# reader = BarcodeReader()
# reader.init_license("t0068lQAAAIR7f27QvcwKBYrOYBcj4MUlaPvljwnmsabNAoUAO8LPyYazLfTtYb6Jw13t0VBd/SZ7JDGWqr1gxODPK/hXw84=;t0068lQAAAJu64RBSURJZKGQibct+QQD9IN2Bew/740XAMILKK3gdO3ln9gWbU5jVicM092asniRPrRt3tODxz62FI6+OJNI=")
# input_file="input (3).jpg"
# output_file="SPAN/"
# def barcodeReader(input_file,output_file):
#     image = input_file
#     boxedImg = output_file

#     count = 0
#     new_array = []
#     bounding_boxes_barcode = []
#     # with Image.open(image) as img:
#     #         draw = ImageDraw.Draw(img)
#     img = cv2.imread(image)
#     while count < 2:

#         if count == 0:
#             settings = reader.get_runtime_settings()
#             settings.barcode_format_ids = EnumBarcodeFormat.BF_ALL
#             # settings.barcode_format_ids_2 = EnumBarcodeFormat_2.BF2_ALL
#             # settings.excepted_barcodes_count = 10
#             reader.update_runtime_settings(settings)

#         if count == 1:
#             settings = reader.get_runtime_settings()
#             settings.barcode_format_ids_2 = EnumBarcodeFormat_2.BF2_ALL
#             reader.update_runtime_settings(settings)
#         try:
#             text_results = reader.decode_file(image)
#             if text_results != None:

                
#                 for text_result in text_results:
#                     if len(text_result.barcode_format_string) == 0:
#                         continue
#                     else:
#                         boxes = text_result.localization_result.localization_points
                        
#                         bounding_boxes_barcode.append(boxes)
#                         expanded_vertices = expand_polygon(boxes, offset=10)
#                         # draw.polygon(expanded_vertices, outline="blue", width=3)
                        
#                         pts = np.array(expanded_vertices, np.int32)
#                         pts = pts.reshape((-1,1,2))
#                         # cv2.polylines(img, [pts], True, (0, 0, 255), thickness=7)
#                         detailVal = re.sub(r'\W+', '', text_result.barcode_text)

#                         new_array.append(
#                         {
#                             "DetailValue": detailVal,
#                             "DetailColor": "blue",
#                             "DetailName": text_result.barcode_format_string,
#                             "Boxes":boxes
#                         })
                   
        
#         except BarcodeReaderError as bre:
#             print(bre)
#         count += 1

#     # img.save(boxedImg)

#     filtered_codes = filter_unique_barcode_texts(new_array)

#     for filtered_code in filtered_codes:
#         expanded_vertices = expand_polygon(filtered_code["Boxes"], offset=10)
                        
#         pts = np.array(expanded_vertices, np.int32)
#         pts = pts.reshape((-1,1,2))
#         cv2.polylines(img, [pts], True, (0, 0, 255), thickness=7)

#     cv2.imwrite(boxedImg, img)

#     return filtered_codes
#     # return new_array


# def expand_polygon(vertices, offset):
#     """
#     Expand the polygon outward using Shapely's buffer method.
#     """
#     # Create a polygon from the vertices
#     poly = Polygon(vertices)

#     # Apply the buffer to expand the polygon (positive offset expands outward)
#     expanded_poly = poly.buffer(offset, join_style=2)  # '2' ensures a mitre joint for sharp corners

#     # Convert the expanded polygon back to a list of vertices
#     expanded_vertices = [(int(x), int(y)) for x, y in expanded_poly.exterior.coords]

#     return expanded_vertices


# def filter_unique_barcode_texts(text_results):
#     unique_text_results = {}
#     for text_result in text_results:
#         # Use the barcode_text as the key to ensure uniqueness
#         if text_result["DetailValue"] not in unique_text_results:
#             unique_text_results[text_result["DetailValue"]] = text_result

#     print("Filtered Barcode Texts: " + str((unique_text_results.values())))
#     # Return the values as a list, containing elements with unique barcode_text values
#     return list(unique_text_results.values())


import os
from dbr import BarcodeReader, EnumBarcodeFormat, EnumBarcodeFormat_2, BarcodeReaderError
import cv2
import re
import numpy as np
from shapely.geometry import Polygon

# Initialize the Barcode Reader with your license
reader = BarcodeReader()
reader.init_license("t0068lQAAAIR7f27QvcwKBYrOYBcj4MUlaPvljwnmsabNAoUAO8LPyYazLfTtYb6Jw13t0VBd/SZ7JDGWqr1gxODPK/hXw84=;t0068lQAAAJu64RBSURJZKGQibct+QQD9IN2Bew/740XAMILKK3gdO3ln9gWbU5jVicM092asniRPrRt3tODxz62FI6+OJNI=")  # Replace "YOUR_LICENSE_KEY" with your actual license key.

# Define the input and output paths
input_file = "pharma3.jpg"  # Path to your input image
output_dir = "SPAN"             # Directory to save output image
output_file = os.path.join(output_dir, "output_image.jpg")


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
