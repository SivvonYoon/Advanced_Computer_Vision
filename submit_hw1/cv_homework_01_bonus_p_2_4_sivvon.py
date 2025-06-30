# 2023 computer_vision_class_homework_01_sivvon
# 전자전기공학부 석사과정 윤시원 (2023000853)
# problem 2-4
import os
import cv2 as cv
import time

# image_dataset_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Capricho Gaudi/'
# image_save_dir = '/home/sivvon/Desktop/CV_Class_HW/repeatability/Capricho Gaudi/'
image_dataset_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Episcopal Gaudi/'
image_save_dir = '/home/sivvon/Desktop/CV_Class_HW/repeatability/Episcopal Gaudi/'

os.makedirs(image_save_dir, exist_ok=True)

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

# harris corner detection
for i in img_path_list:
    img = cv.imread(i)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 01 --> blockSize=2, ksize=3, k=0.04, 0.01
    harris_start_time_01 = time.process_time()
    harris_dst_01 = cv.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)    # threshold 01 -1
    harris_dst_01 = cv.dilate(harris_dst_01, None)
    harris_end_time_01 = time.process_time()
    harris_elapsed_time_01 = harris_end_time_01 - harris_start_time_01
    harris_img_01 = img.copy()
    harris_img_01[harris_dst_01 > 0.01*harris_dst_01.max()] = [0, 255, 255]    # threshold 01 - 2 : change correct pixel color to yellow --> corner point
    # cv.imshow('harris_2_3_0.04_img.jpg', harris_img_01)
    image_name = i.split('/')[-1]
    image_name = image_name.split('.')[0]
    cv.imwrite(image_save_dir+image_name+'_harris_2_3_0.04_0.01_img.jpg', harris_img_01)
    print(f"{image_name} : harris_2_3_0.04_0.01_elapsed_time : {harris_elapsed_time_01}")
    harris_corner_01 = len(harris_img_01[(harris_dst_01 > 0.01*harris_dst_01.max())])
    print(f"number of detected harris corners --> {harris_corner_01}")
    # cv.waitKey(0)
    # 02 --> blockSize=4, ksize=5, k=0.04, 0.02
    harris_start_time_02 = time.process_time()
    harris_dst_02 = cv.cornerHarris(gray_img, blockSize=4, ksize=5, k=0.04)    # threshold 02
    harris_dst_02 = cv.dilate(harris_dst_02, None)
    harris_end_time_02 = time.process_time()
    harris_elapsed_time_02 = harris_end_time_02 - harris_start_time_02
    harris_img_02 = img.copy()
    harris_img_02[harris_dst_02 > 0.02*harris_dst_02.max()] = [0, 255, 255]   # threshold 02 - 2 : 0.01 to 0.02 ( to reduce the corner --> to pick robust corner )
    # cv.imshow('harris_2_3_0.04_img.jpg', harris_img_02)
    cv.imwrite(image_save_dir+image_name+'_harris_4_5_0.04_0.02_img.jpg', harris_img_02)
    print(f"{image_name}: harris_4_5_0.04_0.02_elapsed_time : {harris_elapsed_time_02}")
    harris_corner_02 = len(harris_img_02[(harris_dst_02 > 0.02*harris_dst_02.max())])
    print(f"number of detected harris corners --> {harris_corner_02}")
    # cv.waitKey(0)

