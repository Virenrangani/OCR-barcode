import cv2
from datamatrix import DataMatrix

def detect_and_draw_datamatrix(file_path, output_path):
    try:
        # Load the image in grayscale
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError("Could not load the image. Check the file path.")

        # Detect and decode the Data Matrix
        dm = DataMatrix(file_path)  # Pass the file path to DataMatrix
        result = dm.decode()  # Decode the data matrix

        if not result:
            print("No Data Matrix detected.")
            return

        print("Decoded Data Matrix Content:", result)

        # Load the original image (to draw the bounding box)
        original_image = cv2.imread(file_path)

        # Draw a bounding box around the Data Matrix
        for bbox in dm.bounding_boxes:  # `bounding_boxes` gives coordinates
            x, y, w, h = bbox  # Top-left corner (x, y) and width/height
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save and display the result
        cv2.imwrite(output_path, original_image)
        print(f"Output saved at: {output_path}")

        cv2.imshow("Detected Data Matrix", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

# Input and output file paths
file_path = "data matrix.jpg"  # Replace with your input image file
output_path = "output_with_square.jpg"  # Output image file

# Run the function
detect_and_draw_datamatrix(file_path, output_path)

