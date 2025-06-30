# 2023 computer_vision_class_homework_01_sivvon
# 전자전기공학부 석사과정 윤시원 (2023000853)
# problem 1
import os
import cv2 as cv
import time

img_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Episcopal Gaudi/5660437943_9df2449b9f_o.jpg'
save_dir = '/home/sivvon/Desktop/CV_Class_HW/Episcopal Gaudi/'
# img_dir = '/home/sivvon/Desktop/CV_Class_HW/Image Data/Sleeping Beauty Castle Paris/5521226445_811b025f94_o.jpg'
# save_dir = '/home/sivvon/Desktop/CV_Class_HW/Sleeping Beauty Castle Paris/'

os.makedirs(save_dir, exist_ok=True)

# problem 1-1
# image from datasets
img_from_datasets = cv.imread(img_dir)
img = cv.resize(img_from_datasets, (0,0), fx=0.3, fy=0.3)    # image size is too big --> resize to smaller image(300,300)
cv.imwrite(save_dir+'origin_img.jpg', img)                   
cv.imshow('origin_img', img)                                 
cv.waitKey(0)                                                

# problem 1-2 : 2D Gaussian Filt
# use border type to 'BORDER_REPLICATE' --> to take the nearest pixel value
_2d_gaussian_start_time = time.process_time()
img_filtered = cv.GaussianBlur(img, (5,5), sigmaX=1, borderType=1)
_2d_gaussian_end_time = time.process_time()
_2d_gaussian_elapsed_time = _2d_gaussian_end_time - _2d_gaussian_start_time
cv.imshow('img_filtered', img_filtered)                          
cv.imwrite(save_dir+'gaussian_filtered_img.jpg', img_filtered)   
print(f"2d_gaussian_elapsed_time : {_2d_gaussian_elapsed_time}")
cv.waitKey(0)                                                   

# problem 1-3 : Canny Edge Detector
# grayscale the image
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 01 : threshold --> 50, 100
canny_start_time_01 = time.process_time()
canny_img_01 = cv.Canny(gray_img, threshold1=50, threshold2=100, apertureSize=3, L2gradient=True)
canny_end_time_01 = time.process_time()
canny_elapsed_time_01 = canny_end_time_01 - canny_start_time_01
# 02 : threshold --> 0, 0
canny_start_time_02 = time.process_time()
canny_img_02 = cv.Canny(gray_img, threshold1=0, threshold2=0, apertureSize=3, L2gradient=False)
canny_end_time_02 = time.process_time()
canny_elapsed_time_02 = canny_end_time_02 - canny_start_time_02
# 03 : threshold --> 400, 400, 3, L2gradient:True
canny_start_time_03 = time.process_time()
canny_img_03 = cv.Canny(gray_img, threshold1=400, threshold2=400, apertureSize=3, L2gradient=True)
canny_end_time_03 = time.process_time()
canny_elapsed_time_03 = canny_end_time_03 - canny_start_time_03
# 04 : threshold --> 400, 400, 3, L2gradient:False
canny_start_time_04 = time.process_time()
canny_img_04 = cv.Canny(gray_img, threshold1=400, threshold2=400, apertureSize=3, L2gradient=False)
canny_end_time_04 = time.process_time()
canny_elapsed_time_04 = canny_end_time_04 - canny_start_time_04
# 05 : threshold --> 400, 400, 7, L2gradient:True
canny_start_time_05 = time.process_time()
canny_img_05 = cv.Canny(gray_img, threshold1=400, threshold2=400, apertureSize=7, L2gradient=True)
canny_end_time_05 = time.process_time()
canny_elapsed_time_05 = canny_end_time_05 - canny_start_time_05
# 06 : threshold --> 400, 400, 7, L2gradient:False
canny_start_time_06 = time.process_time()
canny_img_06 = cv.Canny(gray_img, threshold1=400, threshold2=400, apertureSize=7, L2gradient=False)
canny_end_time_06 = time.process_time()
canny_elapsed_time_06 = canny_end_time_06 - canny_start_time_06

# 01 : threshold --> 50, 100, 3, True
cv.imshow('canny_50_100_img.jpg', canny_img_01)
cv.imwrite(save_dir+'canny_50_100_img.jpg', canny_img_01)
print(f"canny_50_100_elapsed_time : {canny_elapsed_time_01}")
cv.waitKey(0)
# 02 : threshold --> 0, 0, 3, True
cv.imshow('canny_0_0_img.jpg', canny_img_02)
cv.imwrite(save_dir+'canny_0_0_img.jpg', canny_img_02)
print(f"canny_0_0_elapsed_time : {canny_elapsed_time_02}")
cv.waitKey(0)
# 03 : threshold --> 400, 400, 3, True
cv.imshow('canny_400_400_3_True_img.jpg', canny_img_03)
cv.imwrite(save_dir+'canny_400_400_3_True_img.jpg', canny_img_03)
print(f"canny_400_400_3_True_elapsed_time : {canny_elapsed_time_03}")
cv.waitKey(0)
# 04 : threshold --> 400, 400, 3, False
cv.imshow('canny_400_400_3_False_img.jpg', canny_img_04)
cv.imwrite(save_dir+'canny_400_400_3_False_img.jpg', canny_img_04)
print(f"canny_400_400_3_False_elapsed_time : {canny_elapsed_time_04}")
cv.waitKey(0)
# 05 : threshold --> 400, 400, 7, True
cv.imshow('canny_400_400_7_True_img.jpg', canny_img_05)
cv.imwrite(save_dir+'canny_400_400_7_True_img.jpg', canny_img_05)
print(f"canny_400_400_7_True_elapsed_time : {canny_elapsed_time_05}")
cv.waitKey(0)
# 04 : threshold --> 400, 400, 7, False
cv.imshow('canny_400_400_7_False_img.jpg', canny_img_06)
cv.imwrite(save_dir+'canny_400_400_7_False_img.jpg', canny_img_06)
print(f"canny_400_400_7_False_elapsed_time : {canny_elapsed_time_06}")
cv.waitKey(0)

# problem 1-4 : bilateral filter
# 01 : default
bilater_start_time_01 = time.process_time()
bilater_img_01 = cv.bilateralFilter(img, -1, sigmaColor=1, sigmaSpace=1, dst=None, borderType=None)
bilater_end_time_01 = time.process_time()
bilater_elapsed_time_01 = bilater_end_time_01 - bilater_start_time_01
# 01 : default output
cv.imshow('bilater_default_param_filtered_img.jpg', bilater_img_01)
cv.imwrite(save_dir+'bilater_default_param_filtered_img.jpg', bilater_img_01)
print(f"bilater_default_param_elapsed_time : {bilater_elapsed_time_01}")
cv.waitKey(0)
# 02 : tuned --> BORDER TYPE : 1
bilater_start_time_02 = time.process_time()
bilater_img_02 = cv.bilateralFilter(img, 50, sigmaColor=30, sigmaSpace=30, dst=None, borderType=1)
bilater_end_time_02 = time.process_time()
bilater_elapsed_time_02 = bilater_end_time_02 - bilater_start_time_02
# 02 : tuned output --> BORDER TYPE :1
cv.imshow('bilater_img.jpg', bilater_img_02)
cv.imwrite(save_dir+'bilater_img.jpg', bilater_img_02)
print(f"bilater_elapsed_time : {bilater_elapsed_time_02}")
cv.waitKey(0)