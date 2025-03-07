

import cv2
import numpy as np

def resize_image(image, scale_x=0.2, scale_y=0.2, interpolation=cv2.INTER_AREA):
    """
    Resizes an image using cv2.resize with INTER_AREA for high-quality downscaling.

    Args:
        image (numpy.ndarray): The input image.
        scale_x (float): Scaling factor along the x-axis.
        scale_y (float): Scaling factor along the y-axis.
        interpolation (int): Interpolation method (default: cv2.INTER_AREA).

    Returns:
        numpy.ndarray: The resized image.
    """
    return cv2.resize(image, (0, 0), fx=scale_x, fy=scale_y, interpolation=interpolation)

def sift_detect_and_compute(image_path, scale_x=0.5, scale_y=0.5):
    """
    Detects and computes SIFT keypoints and descriptors for a given image.

    Args:
        image_path (str): The path to the input image.
        scale_x (float): Scaling factor for resizing (default: 0.5).
        scale_y (float): Scaling factor for resizing (default: 0.5).

    Returns:
        tuple: A tuple containing the keypoints, descriptors, and resized image.
               Returns (None, None, None) if the image cannot be read.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None, None

    # Resize using INTER_AREA for high-quality downscaling
    img = resize_image(img, scale_x, scale_y, interpolation=cv2.INTER_AREA)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    return keypoints, descriptors, img

def match_features(descriptors1, descriptors2):
    """
    Matches SIFT descriptors using a brute-force matcher with a ratio test.

    Args:
        descriptors1 (np.ndarray): Descriptors from the first image.
        descriptors2 (np.ndarray): Descriptors from the second image.

    Returns:
        list: A list of good matches (DMatch objects). Returns None if no matches.
    """
    if descriptors1 is None or descriptors2 is None:
        print("Error: One or both descriptor sets are None.")
        return None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good_matches

def find_homography(template_keypoints, real_keypoints, good_matches):
    """
    Finds the homography transformation between two images using matched keypoints.

    Args:
        template_keypoints (list): Keypoints of the template image.
        real_keypoints (list): Keypoints of the real image.
        good_matches (list): List of good matches.

    Returns:
        tuple: Homography matrix (H) and the inliers mask.
    """
    if len(good_matches) < 4:
        return None, None

    src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([real_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

def calculate_match_percentage(inliers_mask, good_matches):
    """
    Calculates the percentage of matching keypoints based on inliers.

    Args:
        inliers_mask (np.ndarray): Mask indicating inliers from homography.
        good_matches (list): List of good matches.

    Returns:
        float: Matching percentage.
    """
    if inliers_mask is None:
        return 0.0

    inlier_matches = np.sum(inliers_mask)
    total_matches = len(good_matches)

    return (inlier_matches / total_matches) * 100 if total_matches > 0 else 0.0

def main():
    """
    Main function to demonstrate SIFT feature matching and calculate match percentage.
    """
    template_image_path = 'image2.jpg'
    real_image_path = 'image1.jpg'

    # Detect and compute features for both images
    template_keypoints, template_descriptors, template_img = sift_detect_and_compute(template_image_path, scale_x=0.5, scale_y=0.5)
    real_keypoints, real_descriptors, real_img = sift_detect_and_compute(real_image_path, scale_x=0.3, scale_y=0.3)  # High-quality downscale using INTER_AREA

    if template_keypoints is None or template_descriptors is None or real_keypoints is None or real_descriptors is None:
        print("Error: Could not extract features from one or both images.")
        return

    # Match features
    good_matches = match_features(template_descriptors, real_descriptors)

    if good_matches is None or len(good_matches) < 4:
        print("Not enough good matches found to establish a reliable correspondence.")
        return

    print(f"Number of good matches found: {len(good_matches)}")

    # Find Homography using RANSAC
    H, mask = find_homography(template_keypoints, real_keypoints, good_matches)

    if H is None:
        print("Homography could not be computed.")
        return

    # Calculate match percentage
    match_percentage = calculate_match_percentage(mask, good_matches)
    print(f"Matching Percentage (using inliers): {match_percentage:.2f}%")

    # Draw inlier matches
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    img_matches = cv2.drawMatches(template_img, template_keypoints, real_img, real_keypoints, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Save and display the result
    cv2.imwrite("carton3.jpg", img_matches)

if __name__ == "__main__":
    main()


