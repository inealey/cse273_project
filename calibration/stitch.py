import numpy as np
import cv2
import glob
import os
from datetime import datetime as dt
import matplotlib.pyplot as plt
from math import pi, sin, cos
from scipy.optimize import least_squares

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


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


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


def compute_pose(quats, H, src_center, dst_center, str):
    # Return a 3 x 4 array, with its last column translation vector.
    rotation = np.exp(np.array([[0., -quats[2], quats[1]], [quats[2],
                                                            0., -quats[0]], [-quats[1], quats[0], 0.]], dtype='float32'))
    print(str + " rotation: ")
    print(rotation)
    # v - dst u -src 1x3
    # v = KRK-1u + trans_vec
    ptu = np.array([[src_center[0]], [src_center[1]], [1.]], dtype='float32')
    ptv = np.array([[dst_center[0]], [dst_center[1]], [1.]], dtype='float32')
    invK = np.linalg.inv(K)
    translation = invK @ ptv - rotation @ invK @ ptu
    print(str + " translation: ")
    print(translation)
    return np.concatenate([rotation, translation], 1)


def compute_residual(poses, pairs, len1, len2, len3, len4):
    pair1 = pairs[:, :len1]
    pair2 = pairs[:, len1:len1+len2]
    pair3 = pairs[:, len1+len2: len1+len2+len3]
    pair4 = pairs[:, len1+len2+len3:]
    pairs = [pair1, pair2, pair3, pair4]
    errors_list = []
    for i in range(4):
        pose = poses[i*12:i*12+12].reshape((3, 4))
        pair = pairs[i]
        src_pts = pair[0]
        dst_pts = pair[1]
        ones = np.ones((src_pts.shape[0], 1))
        upts = np.concatenate([src_pts, ones], 1).T
        vpts = np.concatenate([dst_pts, ones], 1).T
        proj_pts = K @ pose[:, :3] @ np.linalg.inv(K) @ upts + pose[:, -1:]
        errors_list.append(vpts.T - proj_pts.T)
        # error_norms = np.linalg.norm(proj_error_vec, axis=0)
        # error_scalar = np.sum(error_norms)
        # # ‘huber’ : rho(z) = z if z <= 1 else 2*z**0.5 - 1
        # residual = residual + error_scalar if error_scalar <= 1 else residual + \
        #     2 * error_scalar ** 0.5 - 1
    return np.concatenate(errors_list, 0).ravel()


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

# find keypoints matches in two images (west --> north)
# north 1, west 2, south 3, east 4
src_dstpairs = []
fpts1, fpts2, _, _ = getSIFTMatchesPair(img_west, img_north, mask)
src_dstpairs.append(np.array([fpts1, fpts2], dtype='float32'))

H12, mask12 = cv2.findHomography(fpts1, fpts2, cv2.RANSAC)
matches_mask = mask12.ravel().tolist()
src_center12 = np.mean(np.array(fpts1[np.bool8(matches_mask)]), 1)
dst_center12 = np.mean(np.array(fpts2[np.bool8(matches_mask)]), 1)

fpts2, fpts3, _, _ = getSIFTMatchesPair(img_south, img_west, mask)
src_dstpairs.append(np.array([fpts2, fpts3], dtype='float32'))

H23, mask23 = cv2.findHomography(fpts2, fpts3, cv2.RANSAC)
matches_mask = mask23.ravel().tolist()
src_center23 = np.mean(np.array(fpts2[np.bool8(matches_mask)]), 1)
dst_center23 = np.mean(np.array(fpts3[np.bool8(matches_mask)]), 1)

fpts3, fpts4, _, _ = getSIFTMatchesPair(img_east, img_south, mask)
src_dstpairs.append(np.array([fpts3, fpts4], dtype='float32'))

H34, mask34 = cv2.findHomography(fpts3, fpts4, cv2.RANSAC)
matches_mask = mask34.ravel().tolist()
src_center34 = np.mean(np.array(fpts3[np.bool8(matches_mask)]), 1)
dst_center34 = np.mean(np.array(fpts4[np.bool8(matches_mask)]), 1)

fpts4, fpts1, _, _ = getSIFTMatchesPair(img_north, img_east, mask)
src_dstpairs.append(np.array([fpts4, fpts1], dtype='float32'))

H41, mask41 = cv2.findHomography(fpts4, fpts1, cv2.RANSAC)
matches_mask = mask41.ravel().tolist()
src_center41 = np.mean(np.array(fpts4[np.bool8(matches_mask)]), 1)
dst_center41 = np.mean(np.array(fpts1[np.bool8(matches_mask)]), 1)

# # Parameters:  alpha1, beta1, theta1, x1, y1, z1, .... , -x1-x2-x3, -y1-y2-y3, -z1-z2-z3
# start_params = [[0, pi / 2, pi / 2, 1e-3, 1e-3, 1e-3], [0, pi / 2,
#                                                         pi / 2, 1e-3, 1e-3, 1e-3], [0, pi / 2, pi / 2, 1e-3, 1e-3, 1e-3]]

# src_pts = np.stack([np.array(fpts1, dtype='float32'), np.array(fpts2, dtype='float32'), np.array(fpts3, dtype='float32'),
#                     np.array(fpts4, dtype='float32')], 0)
# dst_pts = np.stack([np.array(fpts2, dtype='float32'), np.array(fpts3, dtype='float32'), np.array(fpts4, dtype='float32'),
#                     np.array(fpts1, dtype='float32')], 0)
# print(LM(start_params, (src_pts, dst_pts),
#          compute_error, numerical_differentiation))
quat12 = [-float('inf'), -float('inf'), 0.]
quat23 = [-float('inf'), -float('inf'), 0.]
quat34 = [-float('inf'), -float('inf'), 0.]
quat41 = [-float('inf'), -float('inf'), 0.]
pose12 = compute_pose(quat12, H12, src_center12, dst_center12, "12")
pose23 = compute_pose(quat23, H23, src_center23, dst_center23, "23")
pose34 = compute_pose(quat34, H34, src_center34, dst_center34, "34")
pose41 = compute_pose(quat41, H41, src_center41, dst_center41, "41")
poses = [pose12, pose23, pose34, pose41]

l12 = src_dstpairs[0].shape[1]
l23 = src_dstpairs[1].shape[1]
l34 = src_dstpairs[2].shape[1]
l41 = src_dstpairs[3].shape[1]
src_dstpairs = np.concatenate(src_dstpairs, 1)
poses = np.stack(poses, 0).reshape((48,))

res = least_squares(compute_residual, poses, verbose=1, x_scale='jac', ftol=1e-6, method='lm',
                    args=(src_dstpairs, l12, l23, l34, l41))

print("Check whether the circle closes?")
print(res.x[9:12])
print(res.x[21:24])
print(res.x[9:12] + res.x[21:24] + res.x[33:36] + res.x[45:48])
