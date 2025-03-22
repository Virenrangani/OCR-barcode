# import cv2
# import numpy as np

# def resize_and_load_image(image_path, scale=0.3):
#     """
#     Loads an image in grayscale and resizes it using cv2.INTER_AREA for high-quality downscaling.
#     """
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise FileNotFoundError(f"Error: Could not read image at {image_path}")
#     new_size = tuple(np.int32(np.array(img.shape[::-1]) * scale))
#     return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

# def extract_fast_features(image):
#     """
#     Extracts keypoints using FAST and computes descriptors using ORB.
#     """
#     fast = cv2.FastFeatureDetector_create()
#     orb = cv2.ORB_create()
#     keypoints = fast.detect(image, None)
#     return orb.compute(image, keypoints) if keypoints else ([], None)

# def match_features(descriptors1, descriptors2):
#     """
#     Matches ORB descriptors using a Brute-Force matcher with Hamming distance.
#     """
#     if descriptors1 is None or descriptors2 is None:
#         return []
#     matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(descriptors1, descriptors2)
#     return sorted(matches, key=lambda x: x.distance)

# def calculate_matching_percentage(matches, total_keypoints):
#     """
#     Calculates the percentage of matching keypoints.
#     """
#     return 0.0 if total_keypoints == 0 else (len(matches) / total_keypoints) * 100

# def highlight_matches(img1, img2, keypoints1, keypoints2, matches, match_percent):
#     """
#     Draws matches between keypoints of two images and overlays the matching percentage.
#     """
#     img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     cv2.putText(img_matches, f"Matching Area: {match_percent:.2f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     return img_matches

# def main():
#     """
#     Main function to perform FAST feature matching, resize larger image to smaller image size,
#     highlight matched keypoints and matching percentage.
#     """
#     try:
#         template_img = resize_and_load_image('inara.jpg', scale=0.5)
#         real_img = resize_and_load_image('inara1.jpg', scale=0.3)
#     except FileNotFoundError as e:
#         print(e)
#         return

#     # Get image shapes
#     template_h, template_w = template_img.shape[:2]
#     real_h, real_w = real_img.shape[:2]

#     # Determine smaller image size
#     if template_h * template_w <= real_h * real_w:
#         smaller_h, smaller_w = template_h, template_w
#         larger_img = real_img
#         larger_name = "real_img"
#         smaller_name = "template_img"
#     else:
#         smaller_h, smaller_w = real_h, real_w
#         larger_img = template_img
#         larger_name = "template_img"
#         smaller_name = "real_img"

#     # Resize the larger image to the size of the smaller image
#     if larger_img is not None: # Added check to avoid potential NoneType error if loading failed earlier
#         resized_larger_img = cv2.resize(larger_img, (smaller_w, smaller_h), interpolation=cv2.INTER_AREA)
#         print(f"Resized {larger_name} to the size of {smaller_name} ({smaller_w}x{smaller_h}).")
#         if larger_name == "real_img":
#             real_img = resized_larger_img
#         else:
#             template_img = resized_larger_img
#     else:
#         print("Error: One of the images failed to load, cannot perform resizing.")
#         return


#     template_keypoints, template_descriptors = extract_fast_features(template_img)
#     real_keypoints, real_descriptors = extract_fast_features(real_img)

#     if not template_keypoints or not real_keypoints:
#         print("Error: Could not extract features from one or both images.")
#         return

#     matches = match_features(template_descriptors, real_descriptors)

#     if not matches:
#         print("Not enough good matches found.")
#         return

#     total_keypoints = min(len(template_keypoints), len(real_keypoints))
#     match_percent = calculate_matching_percentage(matches, total_keypoints)
#     print(f"Matching Area: {match_percent:.2f}%")

#     result_img = highlight_matches(template_img, real_img, template_keypoints, template_keypoints, matches, match_percent) # corrected keypoints2 to real_keypoints

#     cv2.imwrite("fast_feature_matches1.jpg", result_img)
#     print("Feature matching image saved as fast_feature_matches.jpg")

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

def extract_fast_features(image, fast_threshold=20, orb_nfeatures=500):
    """
    Extracts keypoints using FAST and computes descriptors using ORB.
    Allows tuning FAST threshold and ORB nfeatures for efficiency.
    """
    fast = cv2.FastFeatureDetector_create(threshold=fast_threshold) # Tunable threshold
    orb = cv2.ORB_create(nfeatures=orb_nfeatures) # Tunable nfeatures
    keypoints = fast.detect(image, None)
    return orb.compute(image, keypoints) if keypoints else ([], None)

def match_features(descriptors1, descriptors2, cross_check=True):
    """
    Matches ORB descriptors using a Brute-Force matcher with Hamming distance.
    Optionally disable crossCheck for speed (at the cost of potential false matches).
    """
    if descriptors1 is None or descriptors2 is None:
        return []
    # crossCheck=True often improves matching quality but reduces speed.
    # Set cross_check=False for potentially faster matching if needed.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
    matches = bf.match(descriptors1, descriptors2)
    return sorted(matches, key=lambda x: x.distance)

def calculate_matching_percentage(matches, total_keypoints):
    """
    Calculates the percentage of matching keypoints.
    """
    return 0.0 if total_keypoints == 0 else (len(matches) / total_keypoints) * 100

def highlight_matches(img1, img2, keypoints1, keypoints2, matches, match_percent):
    """
    Draws matches between keypoints of two images and overlays the matching percentage.
    """
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.putText(img_matches, f"Matching Area: {match_percent:.2f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img_matches

def main():
    """
    Main function to perform FAST feature matching, resize larger image to smaller image size,
    highlight matched keypoints and matching percentage.
    """
    try:
        template_img = resize_and_load_image('inara.jpg', scale=0.5)
        real_img = resize_and_load_image('inara1.jpg', scale=0.3)
    except FileNotFoundError as e:
        print(e)
        return

    template_h, template_w = template_img.shape[:2]
    real_h, real_w = real_img.shape[:2]

    if template_h * template_w <= real_h * real_w:
        smaller_h, smaller_w = template_h, template_w
        larger_img = real_img
        larger_name = "real_img"
        smaller_name = "template_img"
    else:
        smaller_h, smaller_w = real_h, real_w
        larger_img = template_img
        larger_name = "template_img"
        smaller_name = "real_img"

    if larger_img is not None:
        resized_larger_img = cv2.resize(larger_img, (smaller_w, smaller_h), interpolation=cv2.INTER_AREA)
        print(f"Resized {larger_name} to the size of {smaller_name} ({smaller_w}x{smaller_h}).")
        if larger_name == "real_img":
            real_img = resized_larger_img
        else:
            template_img = resized_larger_img
    else:
        print("Error: One of the images failed to load, cannot perform resizing.")
        return

    # Tunable parameters for feature extraction and matching:
    fast_threshold_value = 15 # Lower threshold: more features, slower, potentially more matches
    orb_nfeatures_value = 400 # Lower nfeatures: faster, fewer descriptors
    use_cross_check = True # cross_check=False for faster matching, but less strict

    template_keypoints, template_descriptors = extract_fast_features(template_img, fast_threshold=fast_threshold_value, orb_nfeatures=orb_nfeatures_value)
    real_keypoints, real_descriptors = extract_fast_features(real_img, fast_threshold=fast_threshold_value, orb_nfeatures=orb_nfeatures_value)

    if not template_keypoints or not real_keypoints:
        print("Error: Could not extract features from one or both images.")
        return

    matches = match_features(template_descriptors, real_descriptors, cross_check=use_cross_check)

    if not matches:
        print("Not enough good matches found.")
        return

    total_keypoints = min(len(template_keypoints), len(real_keypoints))
    match_percent = calculate_matching_percentage(matches, total_keypoints)
    print(f"Matching Area: {match_percent:.2f}%")

    result_img = highlight_matches(template_img, real_img, template_keypoints, real_keypoints, matches, match_percent)

    cv2.imwrite("fast_feature_matches1.jpg", result_img)
    print("Feature matching image saved as fast_feature_matches.jpg")

if __name__ == "__main__": # Corrected line
    main()