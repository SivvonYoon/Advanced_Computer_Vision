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
    gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create(30)    # threshold : 30
    kps = fast.detect(img, None)
    # print(kps)
    fast_img_01 = img.copy()
    fast_img_01 = cv.drawKeypoints(img, keypoints=kps, outImage=fast_img_01, color=None, flags=None)
    cv.imwrite(save_dir+image_name+'_fast_30_img.jpg', fast_img_01)
    # 02 : threshold 300
    fast_start_time_02 = time.process_time()
    fast_02 = cv.FastFeatureDetector_create(300)    # threshold : 300
    fast_end_time_02 = time.process_time()
    fast_elapsed_time_02 = fast_end_time_02 - fast_start_time_02
    kps_02 = fast_02.detect(img, None)
    fast_img_02 = img.copy()
    fast_img_02 = cv.drawKeypoints(img, keypoints=kps_02, outImage=fast_img_02, color=None, flags=None)
    cv.imwrite(save_dir+image_name+'_fast_300_img.jpg', fast_img_02)
    # 03 : threshold 400
    fast_03 = cv.FastFeatureDetector_create(400)    # threshold : 400
    kps_03 = fast_03.detect(img, None)
    fast_img_03 = img.copy()
    fast_img_03 = cv.drawKeypoints(img, keypoints=kps_03, outImage=fast_img_03, color=None, flags=None)
    cv.imwrite(save_dir+image_name+'_fast_400_img.jpg', fast_img_03)
    # 04 : threshold 350
    fast_start_time_04 = time.process_time()
    fast_04 = cv.FastFeatureDetector_create(350)    # threshold : 400
    fast_end_time_04 = time.process_time()
    fast_elapsed_time_04 = fast_end_time_04 - fast_start_time_04
    kps_04 = fast_04.detect(img, None)
    fast_img_04 = img.copy()
    fast_img_04 = cv.drawKeypoints(img, keypoints=kps_04, outImage=fast_img_04, color=None, flags=None)
    cv.imwrite(save_dir+image_name+'_fast_350_img.jpg', fast_img_04)
    # show numbers of keypoiints
    print(f'FAST : {image_name} threshold  30 --> number of keypoints = {len(kps)}\n')
    print(f'FAST : {image_name} threshold 300 --> number of keypoints = {len(kps_02)}\n')
    print(f'FAST : {image_name} threshold 400 --> number of keypoints = {len(kps_03)}\n')
    print(f'FAST : {image_name} threshold 350 --> number of keypoints = {len(kps_04)}\n')
    print(f"{image_name} : fast_300_elapsed_time : {fast_elapsed_time_02}")
    print(f"{image_name} : fast_350_elapsed_time : {fast_elapsed_time_04}")