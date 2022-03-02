import numpy as np
import cv2
import glob
import os
from datetime import datetime as dt
import matplotlib.pyplot as plt

CHECKERBOARD = (6, 8)

subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
    cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW+cv2.CALIB_RATIONAL_MODEL
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('raw_images/*.jpg')
img = cv2.imread(images[-1])
K = np.load('B036_intrintics.npy')
D = np.load('B036_distortion.npy')


def rectify(image, K, balance=1.0, scale=1.0, dim2=None, dim3=None):
    assert K[2][2] == 1.0
    # dim1 is the dimension of input image to un-distort
    dim1 = image.shape[:2][::-1]
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = (int(dim1[0]*scale), int(dim1[1]*scale))

    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image.
    # OpenCV document failed to make this clear!
    # see: https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, dim2, np.eye(3), balance=balance)

    # scale translation vectors accordingly
    new_K[0][2] *= scale
    new_K[1][2] *= scale

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    rectified_img = cv2.remap(
        image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rectified_img


def getSIFTMatchesPair(img1, img2, mask):
    """
    detects matching points from a pair of images
    using SIFT feature detection and the Brute force descriptor matcher.
    inputs:
        img1: Grayscale image1
        img2: Grayscale image2
    Returns:
        corners1: numpy array that contains matching corners from image1 in 
                    image coordinates(Nx2)
        corners2: ^likewise for image 2
        descriptors1: feature descriptors (image1) for each matched feature in corners1
        descriptors2: ^ likewise for image 2, corners 2
    """
    # init sift object
    sift = cv2.SIFT_create()
    corners1, corners2, descriptors1, descriptors2 = [], [], [], []

    # find image keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, mask)  # 'left' image
    kp2, des2 = sift.detectAndCompute(img2, mask)  # 'right' image

    # use BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # filter poor matches
            im1_idx = m.queryIdx
            im2_idx = m.trainIdx
            x1, y1 = kp1[im1_idx].pt
            x2, y2 = kp2[im2_idx].pt
            corners1.append((x1, y1))
            corners2.append((x2, y2))
            descriptors1.append(des1[im1_idx])
            descriptors2.append(des2[im2_idx])

    corners1 = np.array(corners1, dtype=np.float32)
    corners2 = np.array(corners2, dtype=np.float32)
    descriptors1 = np.array(descriptors1, dtype=np.float32)
    descriptors2 = np.array(descriptors2, dtype=np.float32)

    return corners1, corners2, descriptors1, descriptors2


def plotFeatures(img, features):
    feat_x, feat_y = [], []
    for i in range(len(features)):
        feat_x.append(features[i][0])
        feat_y.append(features[i][1])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img, cmap='gray')
    ax.scatter(feat_x, feat_y)
    plt.show()


def plotFeatureCorr(c1, c2, eudist=5):
    assert len(c1) == len(c2)
    # compute a homography between images
    H, mask = cv2.findHomography(c1, c2, cv2.RANSAC)
    matches_mask = mask.ravel().tolist()
    c1_inliers = c1[np.bool8(matches_mask)]
    new_pts = []
    for pt in list(c1):
        dists = np.float32([(pt[0]-inpt[0])**2+(pt[1]-inpt[1])
                            ** 2 for inpt in list(c1_inliers)])
        if np.count_nonzero(dists < eudist**2) == 0:
            new_pts.append(pt)
    return H, new_pts


SCALE = 1.1  # how "far away" to render rectified image

# load images
img_north = cv2.imread(
    '2_9_2022_test_images/North_2022-02-09_160021_image.jpg')
img_south = cv2.imread(
    '2_9_2022_test_images/South_2022-02-09_160030_image.jpg')
img_east = cv2.imread('2_9_2022_test_images/East_2022-02-09_160225.jpg')
img_west = cv2.imread('2_9_2022_test_images/West_2022-02-09_160239.jpg')
mask = np.ones([img_west.shape[0], img_west.shape[1], 1], dtype='uint8')

# rectify and convert to grayscale
img_north = cv2.cvtColor(
    rectify(img_north, K, scale=SCALE), cv2.COLOR_BGR2GRAY)
img_south = cv2.cvtColor(
    rectify(img_south, K, scale=SCALE), cv2.COLOR_BGR2GRAY)
img_east = cv2.cvtColor(rectify(img_east, K, scale=SCALE), cv2.COLOR_BGR2GRAY)
img_west = cv2.cvtColor(rectify(img_west, K, scale=SCALE), cv2.COLOR_BGR2GRAY)
mask = rectify(mask, K, scale=SCALE)

mask_l = mask.copy()
mask_r = mask.copy()
# only look for SIFT points on the right half of image
mask_l[:, :int(mask.shape[1] / 2):] = 0
# only look for SIFT points on the left half of image
mask_r[:, int(mask.shape[1] / 2):] = 0

# find keypoints matches in two images (west --> north)
c1, c2, d1, d2 = getSIFTMatchesPair(img_west, img_north, mask)
_, mask = cv2.findHomography(c1, c2, cv2.RANSAC)
matches_mask = mask.ravel().tolist()
inliers = np.array(c2[np.bool8(matches_mask)]).reshape((-1, 1, 2))

H_w2n, pts_w = plotFeatureCorr(c1, c2)
proj_ptsw = cv2.perspectiveTransform(
    np.array(pts_w).reshape((-1, 1, 2)), H_w2n)
inliers = np.concatenate([inliers, proj_ptsw], 0)
cw_y, cw_x = [], []
for i in range(len(pts_w)):
    cw_x.append(pts_w[i][0])
    cw_y.append(pts_w[i][1])

c1, c2, d1, d2 = getSIFTMatchesPair(img_east, img_north, mask)

H_e2n, pts_e = plotFeatureCorr(c1, c2)
proj_ptse = cv2.perspectiveTransform(
    np.array(pts_e).reshape((-1, 1, 2)), H_e2n)
inliers = np.concatenate([inliers, proj_ptse], 0)
ce_y, ce_x = [], []
for i in range(len(pts_e)):
    ce_x.append(pts_e[i][0])
    ce_y.append(pts_e[i][1])

c1, c2, d1, d2 = getSIFTMatchesPair(img_south, img_west, mask)

H_s2w, pts_s = plotFeatureCorr(c1, c2)
proj_ptss = cv2.perspectiveTransform(
    np.array(pts_s).reshape((-1, 1, 2)), np.matmul(H_s2w, H_w2n))
inliers = np.concatenate([inliers, proj_ptss], 0)
cs_y, cs_x = [], []
for i in range(len(pts_s)):
    cs_x.append(pts_s[i][0])
    cs_y.append(pts_s[i][1])

# fig, ax = plt.subplots(1, 3, figsize=(12, 6))
# ax[0].imshow(img_west, cmap='gray')
# ax[0].scatter(cw_x, cw_y)
# ax[1].imshow(img_east, cmap='gray')
# ax[1].scatter(ce_x, ce_y)
# ax[2].imshow(img_south, cmap='gray')
# ax[2].scatter(cs_x, cs_y)
# plt.show()

xmin = int(inliers[np.argmin(inliers[:, 0, 0]), 0, 0])
xmax = int(inliers[np.argmax(inliers[:, 0, 0]), 0, 0])
ymin = int(inliers[np.argmin(inliers[:, 0, 1]), 0, 1])
ymax = int(inliers[np.argmax(inliers[:, 0, 1]), 0, 1])

mosaic_pts = np.float32([[0, 0], [xmax-xmin-1, 0], [0, ymax-ymin-1],
                         [xmax-xmin-1, ymax-ymin-1]]).reshape(-1, 1, 2)
org_pts = np.float32([[xmin, ymin], [xmax-1, ymin],
                      [xmin, ymax-1], [xmax-1, ymax-1]]).reshape(-1, 1, 2)
H_mosaic = cv2.getPerspectiveTransform(org_pts, mosaic_pts)
# print("xmin: %d, xmax: %d, ymin: %d, ymax: %d" % xmin % xmax % ymin % ymax)
np.save('H_mosaic.npy', H_mosaic)

panorama = np.zeros((ymax-ymin, xmax-xmin, 3), dtype=np.uint8)

result = cv2.warpPerspective(
    img_north, H_mosaic, (xmax-xmin, ymax-ymin), flags=cv2.INTER_NEAREST)
print(result.shape)
panorama[result != 0] = result[result != 0]

H_west = np.matmul(H_mosaic, H_w2n)
result = cv2.warpPerspective(
    img_west, H_west, (xmax-xmin, ymax-ymin), flags=cv2.INTER_NEAREST)
panorama[result != 0] = result[result != 0]

H_east = np.matmul(H_mosaic, H_e2n)
result = cv2.warpPerspective(
    img_east, H_east, (xmax-xmin, ymax-ymin), flags=cv2.INTER_NEAREST)
panorama[result != 0] = result[result != 0]

H_south = np.matmul(H_mosaic, np.matmul(H_s2w, H_w2n))
result = cv2.warpPerspective(
    img_south, H_south, (xmax-xmin, ymax-ymin), flags=cv2.INTER_NEAREST)
panorama[result != 0] = result[result != 0]

result_image = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
plt.imshow(result_image)
plt.show()
plt.save('mosaic.png', )
