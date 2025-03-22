

# import cv2
# import numpy as np

# def resize_image(image, width=200, height=500, interpolation=cv2.INTER_AREA):
#     """
#     Resizes an image to a fixed width and height.
#     """
#     return cv2.resize(image, (width, height), interpolation=interpolation)

# def shi_tomasi_detect_and_compute(image_path, width=200, height=500, max_corners=500, quality_level=0.01, min_distance=10):
#     """
#     Detects keypoints using Shi-Tomasi corner detection and computes ORB descriptors.
#     """
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None, None, None

#     img = resize_image(img, width, height, interpolation=cv2.INTER_AREA)

#     # Detect corners using Shi-Tomasi
#     corners = cv2.goodFeaturesToTrack(img, max_corners, quality_level, min_distance)
#     keypoints = []
#     if corners is not None:
#         keypoints = [cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=10) for pt in corners]
    
#     # If no corners were found, fallback to ORB keypoint detection
#     orb = cv2.ORB_create()
#     if not keypoints:
#         keypoints = orb.detect(img, None)
    
#     # Compute descriptors using ORB
#     keypoints, descriptors = orb.compute(img, keypoints)
#     return keypoints, descriptors, img

# def match_features(descriptors1, descriptors2):
#     """
#     Matches ORB descriptors using a brute-force matcher with a ratio test.
#     """
#     if descriptors1 is None or descriptors2 is None:
#         print("Error: One or both descriptor sets are None.")
#         return None

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(descriptors1, descriptors2)
#     matches = sorted(matches, key=lambda x: x.distance)
#     return matches

# def find_homography(template_keypoints, real_keypoints, good_matches):
#     """
#     Finds the homography transformation between two images using matched keypoints.
#     """
#     if len(good_matches) < 4:
#         return None, None

#     src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#     dst_pts = np.float32([real_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#     H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#     return H, mask

# def calculate_match_percentage(inliers_mask, good_matches):
#     """
#     Calculates the percentage of matching keypoints based on inliers.
#     """
#     if inliers_mask is None:
#         return 0.0

#     inlier_matches = np.sum(inliers_mask)
#     total_matches = len(good_matches)

#     return (inlier_matches / total_matches) * 100 if total_matches > 0 else 0.0

# def main():
#     """
#     Main function to demonstrate Shi-Tomasi + ORB feature matching and calculate match percentage.
#     """
#     template_image_path = 'inara.jpg'
#     real_image_path = 'inara1.jpg'

#     # Detect and compute features for both images
#     template_keypoints, template_descriptors, template_img = shi_tomasi_detect_and_compute(template_image_path)
#     real_keypoints, real_descriptors, real_img = shi_tomasi_detect_and_compute(real_image_path)

#     if template_keypoints is None or template_descriptors is None or real_keypoints is None or real_descriptors is None:
#         print("Error: Could not extract features from one or both images.")
#         return

#     # Match features
#     good_matches = match_features(template_descriptors, real_descriptors)

#     if good_matches is None or len(good_matches) < 4:
#         print("Not enough good matches found to establish a reliable correspondence.")
#         return

#     print(f"Number of good matches found: {len(good_matches)}")

#     # Find Homography using RANSAC
#     H, mask = find_homography(template_keypoints, real_keypoints, good_matches)

#     if H is None:
#         print("Homography could not be computed.")
#         return

#     # Calculate match percentage
#     match_percentage = calculate_match_percentage(mask, good_matches)
#     print(f"Matching Percentage (using inliers): {match_percentage:.2f}%")

#     # Draw inlier matches
#     inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
#     img_matches = cv2.drawMatches(template_img, template_keypoints, real_img, real_keypoints, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#     # Save and display the result
#     cv2.imwrite("templet-matching-image.jpg", img_matches)

# if __name__ == "__main__":
#     main()









import cv2
import numpy as np

def resize_image(image, target_width, target_height, interpolation=cv2.INTER_AREA):
    """
    Resizes an image to the given target width and height.
    """
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)

def get_image_size(image_path):
    """
    Retrieves the dimensions of an image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    return img.shape[1], img.shape[0]

def shi_tomasi_detect_and_compute(image_path, width, height, max_corners=500, quality_level=0.01, min_distance=10):
    """
    Detects keypoints using Shi-Tomasi corner detection and computes ORB descriptors.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None, None

    img = resize_image(img, width, height, interpolation=cv2.INTER_AREA)

    # Detect corners using Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(img, max_corners, quality_level, min_distance)
    keypoints = []
    if corners is not None:
        keypoints = [cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=10) for pt in corners]
    
    # If no corners were found, fallback to ORB keypoint detection
    orb = cv2.ORB_create()
    if not keypoints:
        keypoints = orb.detect(img, None)
    
    # Compute descriptors using ORB
    keypoints, descriptors = orb.compute(img, keypoints)
    return keypoints, descriptors, img

def match_features(descriptors1, descriptors2):
    """
    Matches ORB descriptors using a brute-force matcher with a ratio test.
    """
    if descriptors1 is None or descriptors2 is None:
        print("Error: One or both descriptor sets are None.")
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def find_homography(template_keypoints, real_keypoints, good_matches):
    """
    Finds the homography transformation between two images using matched keypoints.
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
    """
    if inliers_mask is None:
        return 0.0

    inlier_matches = np.sum(inliers_mask)
    total_matches = len(good_matches)

    return (inlier_matches / total_matches) * 100 if total_matches > 0 else 0.0

def main():
    """
    Main function to demonstrate Shi-Tomasi + ORB feature matching and calculate match percentage.
    """
    template_image_path = 'inara.jpg'
    real_image_path = 'inara1.jpg'

    # Get image sizes
    template_width, template_height = get_image_size(template_image_path)
    real_width, real_height = get_image_size(real_image_path)

    if template_width is None or real_width is None:
        print("Error: Could not read one or both images.")
        return

    # Determine the smaller image dimensions
    target_width = min(template_width, real_width)
    target_height = min(template_height, real_height)

    # Detect and compute features for both images
    template_keypoints, template_descriptors, template_img = shi_tomasi_detect_and_compute(template_image_path, target_width, target_height)
    real_keypoints, real_descriptors, real_img = shi_tomasi_detect_and_compute(real_image_path, target_width, target_height)

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
    cv2.imwrite("templet-matching-SHI-TOMASI.jpg", img_matches)

if __name__ == "__main__":
    main()