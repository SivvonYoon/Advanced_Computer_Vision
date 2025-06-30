import cv2
import numpy as np
import os

dir = '/home/sivvon/Desktop/CV_Class_HW/CV_HW5/bookCovers/'
query_path = '/home/sivvon/Desktop/CV_Class_HW/CV_HW5/bookCovers/queries/'
save_dir = '/home/sivvon/Desktop/CV_Class_HW/cv_hw5_results/'

os.makedirs(save_dir,exist_ok=True)

orb = cv2.ORB_create(
    nfeatures=2000,
    scaleFactor=1.4,
    nlevels=8,
    edgeThreshold=31,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=20,
)

image_index = {}
training_descriptors = []

# image index
k=0
t=0
for i in range(1, 59):
    # image load
    img_path = dir+f'book{i}.jpg'
    img_name = f'book{i}.jpg'
    # print(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # feature
    keypoints, descriptors = orb.detectAndCompute(img, None)
    descriptors = np.float32(descriptors)
    
    training_descriptors.append(descriptors)

# print(training_descriptors[0])

# make visual vocabulary
vocabularySize = 60
descriptors = np.concatenate(training_descriptors)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
attempts = 5
flags = cv2.KMEANS_RANDOM_CENTERS

_, labels, vocabulary = cv2.kmeans(descriptors, vocabularySize, None, criteria, attempts, flags)

# Save the visual vocabulary to a file
fs = cv2.FileStorage(save_dir+"visual_vocabulary.xml", cv2.FILE_STORAGE_WRITE)
fs.write("vocabulary", vocabulary)
fs.release()