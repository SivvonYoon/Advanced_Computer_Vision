# 2023 computer_vision_class_homework_01_sivvon
# 전자전기공학부 석사과정 윤시원 (2023000853)
# problem 2
# new project file
import os
import cv2 as cv
import time

# read the same img file I selected in Problem 1's project file code
image_dataset_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Episcopal Gaudi/'
save_dir = '/home/sivvon/Desktop/CV_Class_HW/repeatability/Episcopal Gaudi/'
# img_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Sleeping Beauty Castle Paris/5521226445_811b025f94_o.jpg'
# save_dir = '/home/sivvon/Desktop/CV_Class_HW/Sleeping Beauty Castle Paris_toshow/'

# just make directory to save different images to different direxctories
os.makedirs(save_dir, exist_ok=True)

# make image directories' list from datasets --> to check the repeatability of detectors (harris vs. FAST)

possible_img_extension = ['.jpg', '.jpeg', '.JPG', '.bmp', '.png']
img_path_list = []      # reset is crucial
for (root, dirs, files) in os.walk(image_dataset_dir):
    if len(files) > 0:
        for file_name in files:
            if os.path.splitext(file_name)[1] in possible_img_extension:
                img_path = root + '/' + file_name

                img_path = img_path.replace('\\', '/')
                img_path_list.append(img_path)


# problem 2-2
# choose FAST corner detector
# 01 : threshold 30
for i in img_path_list:
    image_name = i.split('/')[-1]
    image_name = image_name.split('.')[0]
    img = cv.imread(i)
    img = cv.resize(img, (0,0), fx=0.3, fy=0.3)
    sift = cv.SIFT_create()
    kps, desc = sift.detectAndCompute(img, None)
    sift_img_01 = img.copy()
    cv.imshow('test', sift_img_01)
    sift_img_01 = cv.drawKeypoints(img, keypoints=kps, outImage=sift_img_01, color=None, flags=None)
    cv.imwrite(save_dir+image_name+'_sift_img.jpg', sift_img_01)
    print(f'sift --> {image_name}: number of keypoints = {len(kps)}\n')