

# import cv2
# import numpy as np

# def resize_and_load_image(image_path, scale=0.3):
#     """
#     Loads an image in grayscale and resizes it using cv2.INTER_AREA for high-quality downscaling.

#     Args:
#         image_path (str): Path to the input image.
#         scale (float): Scaling factor (default: 0.3 for 30% of original size).

#     Returns:
#         numpy.ndarray: The resized grayscale image.
#     """
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None

#     new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
#     return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

# def extract_sift_features(image):
#     """
#     Extracts SIFT keypoints and descriptors from an image.

#     Args:
#         image (numpy.ndarray): The input grayscale image.

#     Returns:
#         tuple: (keypoints, descriptors), or (None, None) if extraction fails.
#     """
#     if image is None:
#         return None, None

#     sift = cv2.SIFT_create()
#     return sift.detectAndCompute(image, None)

# def match_features(descriptors1, descriptors2):
#     """
#     Matches SIFT descriptors using FLANN-based matcher with a ratio test.

#     Args:
#         descriptors1 (np.ndarray): Descriptors from the first image.
#         descriptors2 (np.ndarray): Descriptors from the second image.

#     Returns:
#         list: A list of good matches (DMatch objects).
#     """
#     if descriptors1 is None or descriptors2 is None:
#         return None

#     index_params = dict(algorithm=1, trees=5)  # KDTree for SIFT
#     search_params = dict(checks=50)

#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(descriptors1, descriptors2, k=2)

#     return [m for m, n in matches if m.distance < 0.75 * n.distance]

# def find_homography(template_keypoints, real_keypoints, good_matches):
#     """
#     Computes the homography transformation using RANSAC.

#     Args:
#         template_keypoints (list): Keypoints from the template image.
#         real_keypoints (list): Keypoints from the real image.
#         good_matches (list): List of good matches.

#     Returns:
#         tuple: (Homography matrix, inliers mask).
#     """
#     if len(good_matches) < 4:
#         return None, None

#     src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([real_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#     return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# def calculate_non_matching_percentage(mask, good_matches):
#     """
#     Calculates the percentage of non-matching keypoints (outliers).

#     Args:
#         mask (np.ndarray): Inliers mask from homography.
#         good_matches (list): List of good matches.

#     Returns:
#         float: Percentage of non-matching keypoints.
#     """
#     if mask is None:
#         return 100.0  # If no homography found, assume 100% mismatch

#     inliers = np.sum(mask)  # Count inliers
#     total_matches = len(good_matches)
#     outliers = total_matches - inliers  # Non-matching points

#     return (outliers / total_matches) * 100 if total_matches > 0 else 100.0

# def highlight_non_matching_area(img1, img2, template_keypoints, real_keypoints, good_matches, mask, non_match_percent):
#     """
#     Highlights non-matching areas (outliers) and overlays the mismatch percentage.

#     Args:
#         img1 (numpy.ndarray): The first image (template).
#         img2 (numpy.ndarray): The second image (real image).
#         template_keypoints (list): Keypoints from the template image.
#         real_keypoints (list): Keypoints from the real image.
#         good_matches (list): List of good matches.
#         mask (numpy.ndarray): Inliers mask from the homography calculation.
#         non_match_percent (float): Percentage of non-matching areas.

#     Returns:
#         numpy.ndarray: Image with non-matching areas highlighted.
#     """
#     img_matches = cv2.drawMatches(img1, template_keypoints, img2, real_keypoints, good_matches, None, 
#                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#     # Draw non-matching points in red
#     outlier_matches = [good_matches[i] for i in range(len(good_matches)) if not mask[i]]
    
#     for match in outlier_matches:
#         img1_point = tuple(np.int32(template_keypoints[match.queryIdx].pt))
#         img2_point = tuple(np.int32(real_keypoints[match.trainIdx].pt))
        
#         cv2.circle(img_matches, img1_point, 5, (0, 0, 255), 2)  # Red circle in template
#         cv2.circle(img_matches, img2_point, 5, (0, 0, 255), 2)  # Red circle in real image

#     # Overlay mismatch percentage on the image
#     text = f"Non-Matching Area: {non_match_percent:.2f}%"
#     cv2.putText(img_matches, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     return img_matches

# def main():
#     """
#     Main function to perform SIFT feature matching, calculate non-matching percentage,
#     and highlight non-matching areas.
#     """
#     template_path = 'image2.jpg'
#     real_path = 'image1.jpg'

#     # Load and resize images
#     template_img = resize_and_load_image(template_path, scale=0.5)
#     real_img = resize_and_load_image(real_path, scale=0.3)

#     # Extract features
#     template_keypoints, template_descriptors = extract_sift_features(template_img)
#     real_keypoints, real_descriptors = extract_sift_features(real_img)

#     if template_keypoints is None or real_keypoints is None:
#         print("Error: Could not extract features from one or both images.")
#         return

#     # Match features
#     good_matches = match_features(template_descriptors, real_descriptors)

#     if not good_matches or len(good_matches) < 4:
#         print("Not enough good matches found.")
#         return

#     # Compute Homography
#     H, mask = find_homography(template_keypoints, real_keypoints, good_matches)

#     if H is None:
#         print("Homography could not be computed.")
#         return

#     # Calculate non-matching percentage
#     non_match_percent = calculate_non_matching_percentage(mask, good_matches)
#     print(f"Non-Matching Area: {non_match_percent:.2f}%")

#     # Highlight non-matching areas
#     result_img = highlight_non_matching_area(template_img, real_img, template_keypoints, real_keypoints, good_matches, mask, non_match_percent)

#     # Save output
#     cv2.imwrite("non_matching_highlighted.jpg", result_img)

# if __name__ == "__main__":
#     main()




import cv2
import numpy as np

def resize_and_load_image(image_path, scale=0.3):
    """
    Loads an image in grayscale and resizes it using cv2.INTER_AREA for high-quality downscaling.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Error: Could not read image at {image_path}")
    new_size = tuple(np.int32(np.array(img.shape[::-1]) * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

def extract_fast_features(image):
    """
    Extracts keypoints using FAST and computes descriptors using ORB.
    """
    fast = cv2.FastFeatureDetector_create()
    orb = cv2.ORB_create()
    keypoints = fast.detect(image, None)
    return orb.compute(image, keypoints) if keypoints else ([], None)

def match_features(descriptors1, descriptors2):
    """
    Matches ORB descriptors using a Brute-Force matcher with Hamming distance.
    """
    if descriptors1 is None or descriptors2 is None:
        return []
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(descriptors1, descriptors2)
    return sorted(matches, key=lambda x: x.distance)

def calculate_non_matching_percentage(matches, total_keypoints):
    """
    Calculates the percentage of non-matching keypoints.
    """
    return 100.0 if total_keypoints == 0 else (1 - len(matches) / total_keypoints) * 100

def highlight_matches(img1, img2, keypoints1, keypoints2, matches, non_match_percent):
    """
    Draws matches between keypoints of two images and overlays the non-matching percentage.
    """
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.putText(img_matches, f"Non-Matching Area: {non_match_percent:.2f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img_matches

def main():
    """
    Main function to perform FAST feature matching and highlight matched keypoints.
    """
    try:
        template_img = resize_and_load_image('image2.jpg', scale=0.5)
        real_img = resize_and_load_image('image4.jpg', scale=0.3)
    except FileNotFoundError as e:
        print(e)
        return

    template_keypoints, template_descriptors = extract_fast_features(template_img)
    real_keypoints, real_descriptors = extract_fast_features(real_img)

    if not template_keypoints or not real_keypoints:
        print("Error: Could not extract features from one or both images.")
        return

    matches = match_features(template_descriptors, real_descriptors)

    if not matches:
        print("Not enough good matches found.")
        return
    
    total_keypoints = min(len(template_keypoints), len(real_keypoints))
    non_match_percent = calculate_non_matching_percentage(matches, total_keypoints)
    print(f"Non-Matching Area: {non_match_percent:.2f}%")

    result_img = highlight_matches(template_img, real_img, template_keypoints, real_keypoints, matches, non_match_percent)

    cv2.imwrite("fast_feature_matches.jpg", result_img)
    print("Feature matching image saved as fast_feature_matches.jpg")

if __name__ == "__main__":
    main()
