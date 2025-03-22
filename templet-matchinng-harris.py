import cv2
import numpy as np

def resize_image(image, target_size, interpolation=cv2.INTER_AREA):
    """
    Resizes an image to match the target size.
    """
    return cv2.resize(image, (target_size[1], target_size[0]), interpolation=interpolation)

def get_image_size(image_path):
    """
    Reads an image and returns its size (height, width).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.shape[:2]  # (height, width)

def harris_corner_detection(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """
    Detects Harris corners in an image.
    """
    gray = np.float32(image)
    harris_response = cv2.cornerHarris(gray, block_size, ksize, k)
    harris_response = cv2.dilate(harris_response, None)
    keypoints = np.argwhere(harris_response > threshold * harris_response.max())
    keypoints = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 1) for pt in keypoints]
    return keypoints

def compute_descriptors(image, keypoints):
    """
    Computes ORB descriptors for given keypoints.
    """
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.compute(image, keypoints)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    """
    Matches descriptors using a brute-force matcher with a ratio test.
    """
    if descriptors1 is None or descriptors2 is None:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    return sorted(matches, key=lambda x: x.distance)

def main():
    """Main function to perform Harris Corner Detection and feature matching."""
    template_image_path = 'inara.jpg'
    real_image_path = 'inara1.jpg'
    
    # Get sizes of both images
    template_size = get_image_size(template_image_path)
    real_size = get_image_size(real_image_path)
    
    if template_size is None or real_size is None:
        print("Error: Could not read one or both images.")
        return
    
    # Determine the smaller image size
    target_size = template_size if template_size[0] * template_size[1] < real_size[0] * real_size[1] else real_size
    
    # Read images and resize
    template_img = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    real_img = cv2.imread(real_image_path, cv2.IMREAD_GRAYSCALE)
    
    template_img = resize_image(template_img, target_size)
    real_img = resize_image(real_img, target_size)
    
    # Harris corner detection
    template_keypoints = harris_corner_detection(template_img)
    real_keypoints = harris_corner_detection(real_img)
    
    # Compute ORB descriptors
    template_keypoints, template_descriptors = compute_descriptors(template_img, template_keypoints)
    real_keypoints, real_descriptors = compute_descriptors(real_img, real_keypoints)
    
    if template_descriptors is None or real_descriptors is None:
        print("Error: Could not compute descriptors.")
        return
    
    # Match features
    good_matches = match_features(template_descriptors, real_descriptors)
    if good_matches is None or len(good_matches) < 4:
        print("Not enough good matches found.")
        return
    
    print(f"Number of good matches found: {len(good_matches)}")
    
    # Draw matches
    img_matches = cv2.drawMatches(template_img, template_keypoints, real_img, real_keypoints, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Save and display results
    cv2.imwrite("harris_matches1.jpg", img_matches)
    
if __name__ == "__main__":
    main()
