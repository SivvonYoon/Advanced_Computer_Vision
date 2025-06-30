import cv2
import os
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
import numpy as np
import time

image_1_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Episcopal Gaudi/2878223423_791b00a2b2_o.jpg'
image_2_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Episcopal Gaudi/3929798778_a7eece5baf_o.jpg'

image_save_dir = '/home/sivvon/Desktop/CV_Class_HW/results/'

img1 = cv2.imread(image_1_dir)
img1 = cv2.resize(img1, (0,0), fx=0.3, fy=0.3)
img2 = cv2.imread(image_2_dir)
img2 = cv2.resize(img2, (0,0), fx=0.3, fy=0.3)

os.makedirs(image_save_dir, exist_ok=True)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
print(kp1)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)     # k=2 --> return top 2

# prof. Kyuman Lee's cv lecture note(06) said threshold is usually 0.8 so I use 0.8
better = []
for m, n in matches:
    if m.distance < 0.8*n.distance:
        better.append(m)

total_matches = len(better)

_pts1 = np.float32([ kp1[m.queryIdx].pt for m in better ]).reshape(-1, 2)
_pts2 = np.float32([ kp2[m.trainIdx].pt for m in better ]).reshape(-1, 2)

# Ransac
start = time.process_time()
_, inliers = ransac((_pts1, _pts2),
                    AffineTransform, 
                    min_samples=8,
                    residual_threshold=8, 
                    max_trials=10000
                    )
end = time.process_time()
time_total = end-start

n_inliers = np.sum(inliers)
inlier_ratio = n_inliers / total_matches
print(f"inlier ratio of (# of inlier / of total matches) is {inlier_ratio}")
print(f"it takes time --> {time_total}")

# print(_pts1[inliers])

inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in _pts1[inliers]]
inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in _pts2[inliers]]
placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
image3 = cv2.drawMatches(img1, inlier_keypoints_left, img2, inlier_keypoints_right, placeholder_matches, None)

cv2.imshow('Matches', image3)
cv2.imwrite(image_save_dir+'sift_ransac_match.png', image3)
cv2.waitKey(0)

# visualize 
img1_sift_ransac = img1.copy()
img2_sift_ransac = img2.copy()
img1_sift_ransac = cv2.drawKeypoints(img1_sift_ransac, keypoints=inlier_keypoints_left, outImage=img1_sift_ransac, color=(255,0,0), flags=None)
img2_sift_ransac = cv2.drawKeypoints(img2_sift_ransac, keypoints=inlier_keypoints_right, outImage=img2_sift_ransac, color=(255,0,0), flags=None)
cv2.imshow('left image inliers', img1_sift_ransac)
cv2.imwrite(image_save_dir+'left_image_inliers.png', img1_sift_ransac)
cv2.waitKey(0)
cv2.imshow('right image inliers', img2_sift_ransac)
cv2.imwrite(image_save_dir+'right_image_inliers.png', img2_sift_ransac)
cv2.waitKey(0)

outliers = np.invert(inliers)
outlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in _pts1[outliers]]
outlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in _pts2[outliers]]
img1_sift_ransac = cv2.drawKeypoints(img1_sift_ransac, keypoints=outlier_keypoints_left, outImage=img1_sift_ransac, color=(0,0,255), flags=None)
img2_sift_ransac = cv2.drawKeypoints(img2_sift_ransac, keypoints=outlier_keypoints_right, outImage=img2_sift_ransac, color=(0,0,255), flags=None)
cv2.imshow('left image inliers and outliers', img1_sift_ransac)
cv2.imwrite(image_save_dir+'left_image_inliers_and_outliers.png', img1_sift_ransac)
cv2.waitKey(0)
cv2.imshow('right image inliers and outliers', img2_sift_ransac)
cv2.imwrite(image_save_dir+'right_image_inliers_and_outliers.png', img2_sift_ransac)
cv2.waitKey(0)