# 2023 computer_vision_class_homework_02_sivvon
# 전자전기공학부 석사과정 윤시원 (2023000853)
import os
import cv2 as cv
import time

image_1_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Episcopal Gaudi/2878223423_791b00a2b2_o.jpg'
image_2_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Episcopal Gaudi/3929798778_a7eece5baf_o.jpg'

image_save_dir = '/home/sivvon/Desktop/CV_Class_HW/results/'

os.makedirs(image_save_dir, exist_ok=True)

orb = cv.ORB_create(
    nfeatures=2000,
    scaleFactor=1.4,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20,
)

img1 = cv.imread(image_1_dir)
img1 = cv.resize(img1, (0,0), fx=0.3, fy=0.3)
img2 = cv.imread(image_2_dir)
img2 = cv.resize(img2, (0,0), fx=0.3, fy=0.3)
img1_name = image_1_dir.split('/')[-1]
img2_name = image_2_dir.split('/')[-1]

img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

kp1, des1 = orb.detectAndCompute(img1_gray, None)
kp2, des2 = orb.detectAndCompute(img2_gray, None)
print(kp1)

img1_orb = img1.copy()
img1_orb = cv.drawKeypoints(img1, keypoints=kp1, outImage=img1_orb, color=None, flags=None)
img2_orb = img2.copy()
img2_orb = cv.drawKeypoints(img2, keypoints=kp2, outImage=img2_orb, color=None, flags=None)

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)    # k=2 --> return top 2

better = []
thres = 0.7
# prof. Kyuman Lee's cv lecture note(06) said threshold is usually 0.8 so I use 0.8
for m, n in matches:
    if m.distance < thres*n.distance:
        better.append([m])

result_orb_matching = cv.drawMatchesKnn(img1, kp1, img2, kp2, better, None, flags=2)

# cv.imshow('image_1_origin', img1)
# cv.waitKey(0)
# cv.imshow('image_2_origin', img2)
# cv.waitKey(0)
# cv.imshow('image_1_orb', img1_orb)
# cv.waitKey(0)
# cv.imshow('image_2_orb', img2_orb)
# cv.waitKey(0)
cv.imshow('match', result_orb_matching)
cv.waitKey(0)

cv.imwrite(image_save_dir+img1_name, img1)
cv.imwrite(image_save_dir+img2_name, img2)
cv.imwrite(image_save_dir+'orb_'+img1_name, img1_orb)
cv.imwrite(image_save_dir+'orb_'+img2_name, img2_orb)
cv.imwrite(image_save_dir+'orb_match.jpg', result_orb_matching)