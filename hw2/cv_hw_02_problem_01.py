# 2023 computer_vision_class_homework_02_sivvon
# 전자전기공학부 석사과정 윤시원 (2023000853)
import os
import cv2 as cv
import time

image_1_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Episcopal Gaudi/2878223423_791b00a2b2_o.jpg'
image_2_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Episcopal Gaudi/3929798778_a7eece5baf_o.jpg'

image_save_dir = '/home/sivvon/Desktop/CV_Class_HW/results/'

os.makedirs(image_save_dir, exist_ok=True)

sift = cv.SIFT_create()

img1 = cv.imread(image_1_dir)
img1 = cv.resize(img1, (0,0), fx=0.3, fy=0.3)
img2 = cv.imread(image_2_dir)
img2 = cv.resize(img2, (0,0), fx=0.3, fy=0.3)
img1_name = image_1_dir.split('/')[-1]
img2_name = image_2_dir.split('/')[-1]

img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

print(kp1)
print(des1)

img1_sift = img1.copy()
img1_sift = cv.drawKeypoints(img1, keypoints=kp1, outImage=img1_sift, color=None, flags=None)
img2_sift = img2.copy()
img2_sift = cv.drawKeypoints(img2, keypoints=kp2, outImage=img2_sift, color=None, flags=None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)    # k=2 --> return top 2

better = []
thres = 0.6
# prof. Kyuman Lee's cv lecture note(06) said threshold is usually 0.8 so I use 0.8
for m, n in matches:
    if m.distance < thres*n.distance:
        better.append([m])

result_sift_matching = cv.drawMatchesKnn(img1, kp1, img2, kp2, better, None, flags=2)

cv.imshow('image_1_origin', img1)
cv.waitKey(0)
cv.imshow('image_2_origin', img2)
cv.waitKey(0)
cv.imshow('image_1_sift', img1_sift)
cv.waitKey(0)
cv.imshow('image_2_sift', img2_sift)
cv.waitKey(0)
cv.imshow('match', result_sift_matching)
cv.waitKey(0)

cv.imwrite(image_save_dir+img1_name, img1)
cv.imwrite(image_save_dir+img2_name, img2)
cv.imwrite(image_save_dir+'sift_'+img1_name, img1_sift)
cv.imwrite(image_save_dir+'sift_'+img2_name, img2_sift)
cv.imwrite(image_save_dir+'sift_match.jpg', result_sift_matching)