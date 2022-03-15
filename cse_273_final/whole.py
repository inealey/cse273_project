import numpy as np
import cv2
import glob
import os
from datetime import datetime as dt
import matplotlib.pyplot as plt
import math
from math import sin, cos, pi, radians, degrees, sqrt, atan2, ceil
import panorama_operations as pano

K = np.load('B036_intrintics.npy')
D = np.load('B036_distortion.npy')

# shape of original image is (2048, 3072, 3)
img_north = cv2.imread('test_images/North_2022-02-09_160021_image.jpg')
img_south = cv2.imread('test_images/South_2022-02-09_160030_image.jpg')
img_east = cv2.imread('test_images/East_2022-02-09_160225.jpg')
img_west = cv2.imread('test_images/West_2022-02-09_160239.jpg')
img_mask = np.ones([2048, 3072, 1], dtype='uint8')

SCALE = 1.0  # how "far away" to render rectified image
BALANCE = 0.0


img_north, _ = pano.rectify(img_north, K, D, scale=SCALE, balance=BALANCE)
img_south, _ = pano.rectify(img_south, K, D, scale=SCALE, balance=BALANCE)
img_east, _ = pano.rectify(img_east, K, D, scale=SCALE, balance=BALANCE)
img_west, _ = pano.rectify(img_west, K, D, scale=SCALE, balance=BALANCE)
img_mask, K_rectified = pano.rectify(
    img_mask, K, D, scale=SCALE, balance=BALANCE)

# update K
K = K_rectified

# img_north_cyl = cv2.imread('cylinder_overlap_300_0.png')
# img_east_cyl = cv2.imread('cylinder_overlap_300_1.png')
# img_south_cyl = cv2.imread('cylinder_overlap_300_2.png')
# img_west_cyl = cv2.imread('cylinder_overlap_300_3.png')
img_north_cyl = pano.cylindricalWarp(img_north.copy(), K)
img_east_cyl = pano.cylindricalWarp(img_east.copy(), K)
img_south_cyl = pano.cylindricalWarp(img_south.copy(), K)
img_west_cyl = pano.cylindricalWarp(img_west.copy(), K)

_, thresh = cv2.threshold(img_north_cyl, 0, 255, cv2.THRESH_BINARY)
thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
x_max_mask = np.max(np.where(np.max(thresh, axis=0) == 255))
x_min_mask = np.min(np.where(np.max(thresh, axis=0) == 255))

img_north_cyl = img_north_cyl.copy()[:, x_min_mask:x_max_mask]
img_east_cyl = img_east_cyl.copy()[:, x_min_mask:x_max_mask]
img_south_cyl = img_south_cyl.copy()[:, x_min_mask:x_max_mask]
img_west_cyl = img_west_cyl.copy()[:, x_min_mask:x_max_mask]

# let just line them up for goodness sake!
# working in the 2d domain on the surface of a unit cylinder
img_h, img_w, channels = img_north_cyl.shape

# image overlap in pixels
overlap = 300
h_overlap = ceil(overlap/2)
h_width = ceil(img_w/2)

# Y and X offsets for each image (refinement)
# problem if y_offset > Y_padding
y_padding = 200
offsets = np.array([[100, 0],
                    [100, 0],
                    [110, 0],
                    [100, -50]])

# (refinement) rotations before compositing
# rotations about each image center in degrees
rots = np.array([0, 0, 0, 0])
R = np.array([cv2.getRotationMatrix2D((img_w/2, img_h/2), r, 1) for r in rots])

# make a canvas, add padding to y
canvas = np.zeros((4, img_h + y_padding, img_w * 4 -
                   overlap * 4, channels), dtype=np.uint8)
canvas_h, canvas_w = canvas.shape[1:-1]

# paint in each camera's contribution on the canvas (unrolled cylinder)
# the first image (at the "ends") we will only translate for now.
# its orientation will be the global reference in terms of rotation
# (see automatic stitching paper section "automatic straightening")
# note that images are not reprojected to cylinder after 2d transformations, so we are introducing distortions here
canvas[0, offsets[0, 0]:canvas_h-(y_padding-offsets[0, 0]), 0:h_width+h_overlap +
       0+offsets[0, 1]] = img_north_cyl[:, h_width-h_overlap-offsets[0, 1]:]
canvas[0, offsets[0, 0]:canvas_h-(y_padding-offsets[0, 0]), canvas.shape[2]-h_width +
       h_overlap+offsets[0, 1]:] = img_north_cyl[:, :h_width-h_overlap-offsets[0, 1]]
canvas[1, offsets[1, 0]:canvas_h-(y_padding-offsets[1, 0]), h_width-h_overlap+offsets[1, 1]:int(
    1.5*img_w)-h_overlap+offsets[1, 1]] = cv2.warpAffine(img_east_cyl, R[1], (img_w, img_h))
canvas[2, offsets[2, 0]:canvas_h-(y_padding-offsets[2, 0]), int(1.5*img_w)-3*h_overlap+offsets[2, 1]:int(
    2.5*img_w)-3*h_overlap+offsets[2, 1]] = cv2.warpAffine(img_south_cyl, R[2], (img_w, img_h))
canvas[3, offsets[3, 0]:canvas_h-(y_padding-offsets[3, 0]), int(2.5*img_w)-5*h_overlap+offsets[3, 1]:int(
    3.5*img_w)-5*h_overlap+offsets[3, 1]] = cv2.warpAffine(img_west_cyl, R[3], (img_w, img_h))

# start with a linear blend
# this method is super slow, steps through every pixel
# at least it makes the blending decision at every pixel very clear
# build on this for multi band...?

# gain compensation: compute g1, g2, g3, g4

linear_blend = np.zeros(canvas.shape[1:], dtype=np.float32)
half_width = canvas.shape[2] // 2

sig1 = 10
sig2 = 0.1
coefficients = []
# equation 1, for image 1.
center_img = canvas[0]
left_img = canvas[-1]
right_img = canvas[1]
left_overlap_cnt = 0
left_sum_pixel = np.zeros((2, ))  # mean of I_ij and I_ji
right_overlap_cnt = 0
right_sum_pixel = np.zeros((2, ))
for i in range(int(canvas.shape[1])):
    for j in range(int(canvas.shape[2])):
        center_pixel = center_img[i, j]
        left_pixel = left_img[i, j]
        if center_pixel.all() == 0:
            continue
        if left_pixel.all() != 0:
            left_sum_pixel[1] += np.linalg.norm(left_pixel) / 255.
            left_sum_pixel[0] += np.linalg.norm(center_pixel) / 255.
            left_overlap_cnt += 1

        right_pixel = right_img[i, j]
        if right_pixel.all() != 0:
            right_sum_pixel[1] += np.linalg.norm(right_pixel) / 255.
            right_sum_pixel[0] += np.linalg.norm(center_pixel) / 255.
            right_overlap_cnt += 1

intensity12 = right_sum_pixel[0] / right_overlap_cnt
intensity21 = right_sum_pixel[1] / right_overlap_cnt

intensity14 = left_sum_pixel[0] / left_overlap_cnt
intensity41 = left_sum_pixel[1] / left_overlap_cnt

main_coeff = left_overlap_cnt * \
    (2 * intensity12 ** 2 / (sig1 ** 2) + 1 / (sig2 ** 2))
main_coeff += right_overlap_cnt * \
    (2 * intensity14 ** 2 / (sig1 ** 2) + 1 / (sig2 ** 2))
coefficients.append(np.array([main_coeff, -right_overlap_cnt * intensity12 * intensity21 / (sig1 ** 2), 0.,
                              -left_overlap_cnt * intensity14 * intensity41 / (sig1 ** 2)], dtype='float32'))
print(coefficients[-1])
# for i in range(int(canvas.shape[1])):
#     for j in range(int(canvas.shape[2])):
#         layers = []
#         for c in range(4):
#             values = canvas[c, i, j, :]

#             if values.all() == 0:
#                 continue
#             else:
#                 layers.append(values)

#         if len(layers) == 1:
#             linear_blend[i, j] = layers[0]
#         elif len(layers) == 2:
#             if j < half_width:

#             linear_blend[i, j] = np.sum(canvas[:, i, j, :], axis=0) / 2
#             linear_blend[i, j, -1] = 255

# linear_blend_8 = np.array(linear_blend, dtype=np.uint8)
# cv2.imwrite('linear_blend1.png', linear_blend_8)
