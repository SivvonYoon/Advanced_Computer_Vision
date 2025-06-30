import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils
import open3d as o3d

def get_disparity(left_img, right_img, patch_rad, min_disp, max_disp):
    h, w = left_img.shape
    disp = np.zeros((h, w), dtype=np.uint8)
    for i in range(patch_rad, h - patch_rad):
        for j in range(patch_rad + max_disp, w - patch_rad):
            ssd = []
            left_patch = left_img[i-patch_rad:i+patch_rad+1, j-patch_rad:j+patch_rad+1].astype(np.float32)
            for d in range(max_disp, min_disp-1, -1):
                if j-d-patch_radius >= 0:
                    right_patch = right_img[i-patch_rad:i+patch_rad+1, j-d-patch_rad:j-d+patch_rad+1].astype(np.float32)
                    ssd.append(np.sum(np.square(left_patch - right_patch)))
            min_ssd = np.min(ssd)
            neg_disp = np.argmin(ssd)
            if np.sum(ssd < 1.5 * min_ssd) < 3 and neg_disp != 0 and neg_disp != len(ssd) - 1:
                disp[i, j] = max_disp - neg_disp

            # plt.plot(disp_candidates)
            # plt.show()
            # exit()
            # print(opt_disp)
        # print(i)
    return disp

def disparity_to_pointcloud(disp_img, K, baseline, left_img):
    h, w = left_img.shape
    p_left = np.asarray([[i, j, 1] for i in range(h) for j in range(w)])
    disp_v = np.reshape(disp_img, (h*w,))
    p_right = p_left.copy()
    p_right[:, 1] -= disp_v

    p_left = p_left[disp_v > 0, :]
    p_right = p_right[disp_v > 0, :]
    # diff = p_right - p_left
    # print(np.sum(diff))
    # print(p_left.shape)
    # print(p_right)

    bv_left = np.matmul(np.linalg.inv(K), p_left.T)
    bv_right = np.matmul(np.linalg.inv(K), p_right.T)

    points = np.zeros(bv_left.shape)
    intensities = np.zeros(bv_left.shape)
    b = np.array([[0], [baseline], [0]])

    for i in range(points.shape[1]):
        A = np.concatenate([bv_left[:, i:i+1], -bv_right[:, i:i+1]], axis=1)
        lambd = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
        points[:, i] = bv_left[:, i] * lambd[0]
        intensities[:, i] = left_img[p_left[i, 0], p_left[i, 1]]

    return points, intensities

def read_K_from_txt(K_file):
    K = np.eye(3)
    with open(K_file) as f:
        for i, line in enumerate(f.readlines()):
            K[i, :] = [float(i) for i in line.strip().split(' ') if i]
    return K

baseline = 0.54
patch_radius = 5
min_disp = 5
max_disp = 50

left_img = cv2.imread("/home/sivvon/Desktop/CV_Class_HW/hw3/CV_HW3/left/000000.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread("/home/sivvon/Desktop/CV_Class_HW/hw3/CV_HW3/right/000000.png", cv2.IMREAD_GRAYSCALE)
h, w = left_img.shape
scale = 1

left_img = cv2.resize(left_img, (w//scale, h//scale))
right_img = cv2.resize(right_img, (w//scale, h//scale))
K = read_K_from_txt("/home/sivvon/Desktop/CV_Class_HW/hw3/CV_HW3/K.txt")
K[:2, :] /= scale

disp = get_disparity(left_img, right_img, patch_radius, min_disp, max_disp) # get_disparity function : for problem 1
print("get disparity done")
# normalized_disp = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# # for i in range(disp.shape[0]):
# #     print(disp[i])

# cv2.imwrite("disparity_image1.jpg", normalized_disp)

P_c, intensities = disparity_to_pointcloud(disp, K, baseline, left_img) # disparity_to_pointcloud function : for problem 2
print(intensities)
P_w = np.mat([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]) * P_c[:, ::10]
# print(P_w.T)
cloud = o3d.geometry.PointCloud()
print("make cloud done")
cloud.points = o3d.utility.Vector3dVector(np.asarray(P_c.T))
cloud.colors = o3d.utility.Vector3dVector(intensities.T/255)
print("get cloud done")

# cv2.imshow("disp", normalized_disp)
o3d.io.write_point_cloud("/media/sivvon/F268-EF5A/output1.ply", cloud)
cv2.waitKey(0)
o3d.visualization.draw_geometries([cloud])