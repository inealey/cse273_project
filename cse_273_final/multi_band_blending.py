#!/usr/bin/python
import numpy as np
import cv2
import sys
import argparse
import math


def preprocess(img1, img2, overlap_w, flag_half):
    if img1.shape[0] != img2.shape[0]:
        print("error: image dimension error")
        sys.exit()
    if overlap_w > img1.shape[1] or overlap_w > img2.shape[1]:
        print("error: overlapped area too large")
        sys.exit()

    w1 = img1.shape[1]
    w2 = img2.shape[1]

    if flag_half:
        shape = np.array(img1.shape)
        shape[1] = w1 / 2 + w2 / 2

        subA = np.zeros(shape)
        subA[:, :w1 / 2 + overlap_w / 2] = img1[:, :w1 / 2 + overlap_w / 2]
        subB = np.zeros(shape)
        subB[:, w1 / 2 - overlap_w / 2:] = img2[:,
                                                w2 - (w2 / 2 + overlap_w / 2):]
        mask = np.zeros(shape)
        mask[:, :w1 / 2] = 1
    else:
        shape = np.array(img1.shape)
        shape[1] = w1 + w2 - overlap_w

        subA = np.zeros(shape)
        subA[:, :w1] = img1
        subB = np.zeros(shape)
        subB[:, w1 - overlap_w:] = img2
        mask = np.zeros(shape)

        mask[:, :math.floor(w1 - overlap_w / 2)] = 1
#         mask[:, :w1 - overlap_w / 2] = 1

    return subA, subB, mask


def ComplexPyramid(img1, img2, mask, leveln):
    LP1 = []
    LP2 = []
    GP = [mask]
    for i in range(leveln - 1):
        next_img1 = cv2.pyrDown(img1)
        next_img2 = cv2.pyrDown(img2)
        next_mask = cv2.pyrDown(mask)
        try:
            LP1.append(img1 - cv2.pyrUp(next_img1, img1.shape[1::-1]))
        except:
            h1, w1, _ = img1.shape
            h2, w2, _ = next_img1.shape
            if h1 != h2 * 2:
                next_img1 = next_img1[:-1]
                img1 = img1[:h2 * 2 - 2]
                next_img2 = next_img2[:-1]
                next_mask = next_mask[:-1]
                img2 = img2[:h2 * 2 - 2]
                mask = mask[:h2 * 2 - 2]
            if w1 != w2 * 2:
                next_img1 = next_img1[:, :-1]
                img1 = img1[:, :w2 * 2 - 2]
                next_img2 = next_img2[:, :-1]
                next_mask = next_mask[:, :-1]
                img2 = img2[:, :w2 * 2 - 2]
                mask = mask[:, :w2 * 2 - 2]
            LP1.append(img1 - cv2.pyrUp(next_img1, img1.shape[1::-1]))
        LP2.append(img2 - cv2.pyrUp(next_img2, img2.shape[1::-1]))

        GP.append(next_mask)
        img1 = next_img1
        img2 = next_img2

    LP1.append(img1)
    LP2.append(img2)
    return LP1, LP2, GP


def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended


def reconstruct(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.pyrUp(img, lev_img.shape[1::-1])
        img += lev_img
    return img


def multi_band_blending(img1, img2, overlap_w, leveln=None, flag_half=False):
    if overlap_w < 0:
        print("error: overlap_w should be a positive integer")
        sys.exit()

    subA, subB, mask = preprocess(img1, img2, overlap_w, flag_half)

    max_leveln = int(np.floor(np.log2(min(img1.shape[0], img1.shape[1],
                                          img2.shape[0], img2.shape[1]))))
    if leveln is None:
        leveln = max_leveln
    if leveln < 1 or leveln > max_leveln:
        print("warning: inappropriate number of leveln")
        leveln = max_leveln

    # Get Gaussian pyramid and Laplacian pyramid
    LPA, LPB, MP = ComplexPyramid(subA, subB, mask, leveln)

    # Blend two Laplacian pyramidspass
    blended = blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = reconstruct(blended)
    result[result > 255] = 255
    result[result < 0] = 0

    return result


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(
        description="A Python implementation of multi-band blending")
    ap.add_argument('-H', '--half', required=False, action='store_true',
                    help="option to blend the left half of the first image \
                          and the right half of the second image")
    ap.add_argument('-f', '--first', default='cylinder_overlap_300_0.png',
                    help="path to the first (left) image")
    ap.add_argument('-s', '--second', default='cylinder_overlap_300_1.png',
                    help="path to the second (right) image")
    ap.add_argument('-o', '--overlap', default=10, type=int,
                    help="width of the overlapped area between two images, \
                          even number recommended")
    ap.add_argument('-l', '--leveln', required=False, type=int,
                    help="number of levels of multi-band blending, \
                          calculated from image size if not provided")
    args = vars(ap.parse_args())

    flag_half = args['half']
    img1 = cv2.imread(args['first'], cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(args['second'], cv2.IMREAD_UNCHANGED)
    overlap_w = args['overlap']
    leveln = args['leveln']

    result = multi_band_blending(img1, img2, overlap_w, leveln, flag_half)
    cv2.imwrite('result.png', result)
