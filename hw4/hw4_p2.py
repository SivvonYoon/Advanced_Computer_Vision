import cv2
import os
import natsort
import numpy as np

img_path_dir = '/home/sivvon/Desktop/CV_Class_HW/KITTI dataset/'
img_save_dir = '/home/sivvon/Desktop/CV_Class_HW/KITTI_dataset_optical_output/'

os.makedirs(img_save_dir,exist_ok=True)

img_path_list = []
img_name_list = []
for (root, dirs, files) in os.walk(img_path_dir):
    if len(files) > 0:
        for file_name in files:
            if os.path.splitext(file_name)[1]=='.png':
                img_path = root + '/' + file_name

                img_path = img_path.replace('\\', '/')
                img_path_list.append(img_path)
                img_name = os.path.splitext(file_name)[0]
                img_name_list.append(img_name)

img_path_list = natsort.natsorted(img_path_list)
img_name_list = natsort.natsorted(img_name_list)

for i in range(len(img_path_list)-1):
    # print(img_name_list[i])
    img1_dir = img_path_list[i]
    img2_dir = img_path_list[i+1]
    img1 = cv2.imread(img1_dir)
    img2 = cv2.imread(img2_dir)

    img1_name = img_name_list[i]
    img2_name = img_name_list[i+1]

    cv2.imwrite(img_save_dir+img1_name+'.png', img1)
    cv2.imwrite(img_save_dir+img2_name+'.png', img2)

    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    _pts1 = cv2.goodFeaturesToTrack(gray_img1, 50, 0.01, 10)
    # print(f"_pts1 is {_pts1}")

    _pts2, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, _pts1, None)

    dst = cv2.addWeighted(img1, 0.6, img2, 0.6, 0)

    for k in range(_pts2.shape[0]):
        if status[k,0] == 0:
            continue

        # print(_pts1[k,0])
        # print(_pts1[k][0])
        # print(tuple(_pts1[k][0].astype(int)))
        cv2.circle(dst, tuple(_pts1[k,0].astype(int)), 4, (0,0,255), 2, cv2.LINE_AA)
        cv2.circle(dst, tuple(_pts2[k,0].astype(int)), 4, (0,0,255), 2, cv2.LINE_AA)
        cv2.arrowedLine(dst, tuple(_pts1[k,0].astype(int)), tuple(_pts2[k,0].astype(int)), (0,255,0), 2)
    
    cv2.imshow('optical flow output of '+img1_name+' and '+img2_name, dst)
    print(img1_name)
    cv2.imwrite(img_save_dir+'klt_of_'+img1_name+'_and_'+img2_name+'.png', dst)
    cv2.waitKey(0)
