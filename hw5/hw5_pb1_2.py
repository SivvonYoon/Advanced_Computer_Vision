import cv2
import numpy as np
import os
from glob import glob
import natsort

dirdir = '/home/sivvon/Desktop/CV_Class_HW/CV_HW5/bookCovers/'
query_path = '/home/sivvon/Desktop/CV_Class_HW/CV_HW5/bookCovers/queries/'
save_dir = '/home/sivvon/Desktop/CV_Class_HW/cv_hw5_results/'

os.makedirs(save_dir,exist_ok=True)

fs = cv2.FileStorage(save_dir+"visual_vocabulary.xml", cv2.FILE_STORAGE_READ)
visualVocabulary = fs.getNode("vocabulary").mat()
fs.release()

queryImages = []
for img in os.listdir(query_path):
    queryImages.append(query_path+img)

queryImages = natsort.natsorted(queryImages)
print(queryImages)

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

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 1)
search_params = dict(checks = 60)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# clustering function
def cluster(des, voca):
    matches = flann.knnMatch(des, voca, k=1)
    clusterAssignment = np.zeros((1, voca.shape[0]), dtype=np.float32)

    for match in matches:
        clusterIndex = match[0].trainIdx
        clusterAssignment[0, clusterIndex] += 1

    return clusterAssignment

for queryIndex, queryImagePath in enumerate(queryImages):
    queryImageName = queryImagePath.split("/")[-1]
    queryImage = cv2.imread(queryImagePath)

    if queryImage is None:
        print("Failed to read query image:", queryImagePath)
        continue

    _, query_descriptor = orb.detectAndCompute(queryImage, None)
    query_descriptor = np.float32(query_descriptor)

    query_clustter_assignment = cluster(query_descriptor, visualVocabulary)

    best_match_score = np.finfo(np.float64).max
    best_match_img = ""

    for bookIndex in range(1,59):
        book_path = dirdir + 'book' + str(bookIndex) + '.jpg'
        book_name = f'book{bookIndex}.jpg'
        book_img = cv2.imread(book_path)

        if book_img is None:
            print(f"book image 를 찾지 못했습니다: {book_path}")
            continue

        _, book_descriptor = orb.detectAndCompute(book_img, None)
        book_descriptor = np.float32(book_descriptor)

        book_cluster_assignment = cluster(book_descriptor, visualVocabulary)

        # calculate match score (origin book - query)
        match_score = cv2.compareHist(query_clustter_assignment, book_cluster_assignment, cv2.HISTCMP_CHISQR)

        # Update the best match
        if match_score < best_match_score:
            best_match_score = match_score
            best_match_img = book_img

    # show query img matches
    cv2.imshow(f'query{queryIndex}', queryImage)
    save_q = save_dir+f'querty{queryIndex}.jpg'
    cv2.imwrite(save_q, queryImage)
    cv2.waitKey(0)
    cv2.imshow(f'best image {queryIndex}', best_match_img)
    save_b = save_dir+f'best image {queryIndex}.jpg'
    cv2.imwrite(save_b, best_match_img)
    cv2.waitKey(0)
    result = cv2.hconcat([queryImage, best_match_img])
    cv2.imshow(f'matching result {queryIndex}', result)
    save_match = save_dir+f'matching result {queryIndex}.jpg'
    cv2.imwrite(save_match, result)
    cv2.waitKey(0)