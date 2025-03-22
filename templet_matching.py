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

def sift_detect_and_compute(image_path, target_size=None):
    """
    Detects and computes SIFT keypoints and descriptors for a given image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None, None

    if target_size:
        img = resize_image(img, target_size)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors, img

def match_features(descriptors1, descriptors2):
    """
    Matches SIFT descriptors using a brute-force matcher with a ratio test.
    """
    if descriptors1 is None or descriptors2 is None:
        return None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good_matches

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
    
    # Detect and compute features with resizing
    template_keypoints, template_descriptors, template_img = sift_detect_and_compute(template_image_path, target_size)
    real_keypoints, real_descriptors, real_img = sift_detect_and_compute(real_image_path, target_size)
    
    if template_keypoints is None or real_keypoints is None:
        print("Error: Could not extract features from one or both images.")
        return
    
    # Match features
    good_matches = match_features(template_descriptors, real_descriptors)
    if good_matches is None or len(good_matches) < 4:
        print("Not enough good matches found to establish a reliable correspondence.")
        return
    
    print(f"Number of good matches found: {len(good_matches)}")
    
    # Find Homography
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
    cv2.imwrite("template-matching-SIFT.jpg", img_matches)

if __name__ == "__main__":
    main()



