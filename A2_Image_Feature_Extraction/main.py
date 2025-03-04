import cv2
import matplotlib.pyplot as plt

# read images in grayscale
image1 = cv2.imread(r"C:\school github projects\CS-4391\A2_Image_Feature_Extraction\car_1.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(r"C:\school github projects\CS-4391\A2_Image_Feature_Extraction\car_2.jpg", cv2.IMREAD_GRAYSCALE)

# initialize sift object with default values
sift_detector = cv2.SIFT_create()

# detect and compute keypoints and descriptors
keypoints1, descriptors1 = sift_detector.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift_detector.detectAndCompute(image2, None)

# initialize brute force matcher
bf_matcher = cv2.BFMatcher()

# brute force matcher to match descriptors and return matches
descriptor_matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)

# apply ratio test to get matches for distance
filtered_matches = []
for match1, match2 in descriptor_matches:
    if match1.distance < 0.75 * match2.distance:
        filtered_matches.append(match1)

# sort matches by distance
filtered_matches = sorted(filtered_matches, key=lambda match: match.distance)

# draw the matches on original input image
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, filtered_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# visualize keypoint matches between both images
plt.figure(figsize=(12, 6))
plt.imshow(matched_image, cmap="gray")
plt.xticks([0, 300, 600, 900, 1200])
plt.yticks([0, 100, 200, 300, 400])
plt.show()
