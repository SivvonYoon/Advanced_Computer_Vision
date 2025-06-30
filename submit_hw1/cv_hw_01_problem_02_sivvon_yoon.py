# 2023 computer_vision_class_homework_01_sivvon
# 전자전기공학부 석사과정 윤시원 (2023000853)
# problem 2
# new project file
import os
import cv2 as cv
import time

# read the same img file I selected in Problem 1's project file code
img_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Episcopal Gaudi/5660437943_9df2449b9f_o.jpg'
save_dir = '/home/sivvon/Desktop/CV_Class_HW/Episcopal Gaudi/'
# img_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Sleeping Beauty Castle Paris/5521226445_811b025f94_o.jpg'
# save_dir = '/home/sivvon/Desktop/CV_Class_HW/Sleeping Beauty Castle Paris_toshow/'

# just make directory to save different images to different direxctories
os.makedirs(save_dir, exist_ok=True)

# problem 2
# problem 2-1
# image from datasets
img_from_datasets = cv.imread(img_dir)
img = cv.resize(img_from_datasets, (0,0), fx=0.3, fy=0.3)    # image size is too big --> resize to smaller image(300,300)
cv.imwrite(save_dir+'origin_img.jpg', img)                   
cv.imshow('origin_img', img)                                 
cv.waitKey(0)                                                

# problem 2-2
# choose FAST corner detector
# 01 : threshold 30
gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
fast = cv.FastFeatureDetector_create(30)    # threshold : 30
kps = fast.detect(img, None)
# print(kps)
fast_img_01 = img.copy()
cv.imshow('test', fast_img_01)
fast_img_01 = cv.drawKeypoints(img, keypoints=kps, outImage=fast_img_01, color=None, flags=None)
cv.imwrite(save_dir+'fast_30_img.jpg', fast_img_01)
cv.imshow('fast_30_img',fast_img_01)
cv.waitKey(0)
# 02 : threshold 300
fast_02 = cv.FastFeatureDetector_create(300)    # threshold : 300
kps_02 = fast_02.detect(img, None)
fast_img_02 = img.copy()
cv.imshow('test2', fast_img_02)
fast_img_02 = cv.drawKeypoints(img, keypoints=kps_02, outImage=fast_img_02, color=None, flags=None)
cv.imwrite(save_dir+'fast_300_img.jpg', fast_img_02)
cv.imshow('fast_300_img',fast_img_02)
cv.waitKey(0)
# 03 : threshold 400
fast_03 = cv.FastFeatureDetector_create(400)    # threshold : 400
kps_03 = fast_03.detect(img, None)
fast_img_03 = img.copy()
fast_img_03 = cv.drawKeypoints(img, keypoints=kps_03, outImage=fast_img_03, color=None, flags=None)
cv.imwrite(save_dir+'fast_400_img.jpg', fast_img_03)
cv.imshow('fast_400_img',fast_img_03)
cv.waitKey(0)
# 04 : threshold 350
fast_04 = cv.FastFeatureDetector_create(350)    # threshold : 400
kps_04 = fast_04.detect(img, None)
fast_img_04 = img.copy()
fast_img_04 = cv.drawKeypoints(img, keypoints=kps_04, outImage=fast_img_04, color=None, flags=None)
cv.imwrite(save_dir+'fast_350_img.jpg', fast_img_04)
cv.imshow('fast_350_img',fast_img_04)
cv.waitKey(0)
# show numbers of keypoiints
print(f'FAST :threshold  30 --> number of keypoints = {len(kps)}\n')
print(f'FAST :threshold 300 --> number of keypoints = {len(kps_02)}\n')
print(f'FAST :threshold 400 --> number of keypoints = {len(kps_03)}\n')
print(f'FAST :threshold 350 --> number of keypoints = {len(kps_04)}\n')

# problem 2-3
# SIFT blob detector
sift = cv.SIFT_create()
kps, desc = sift.detectAndCompute(img, None)
sift_img_01 = img.copy()
cv.imshow('test', sift_img_01)
sift_img_01 = cv.drawKeypoints(img, keypoints=kps, outImage=sift_img_01, color=None, flags=None)
cv.imwrite(save_dir+'sift_img.jpg', sift_img_01)
cv.imshow('sift_img',sift_img_01)
cv.waitKey(0)
print(f'sift --> number of keypoints = {len(kps)}\n')