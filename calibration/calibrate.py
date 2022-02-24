6import numpy as np
import cv2 as cv
import glob
import os
from datetime import datetime as dt

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('raw_images/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (8,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (8,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(100)
cv.destroyAllWindows()

#print(imgpoints)
#print(imgpoints[0].shape)
#print(objpoints)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# write out camera calibration
ts = dt.now().strftime("%d-%b-%Y_%H-%M-%S")
os.mkdir('camera/' + ts)
os.mkdir('camera/' + ts + '/imgpts')
os.mkdir('camera/' + ts + '/rvecs')
os.mkdir('camera/' + ts + '/tvecs')

i = 0
for e in imgpoints:
    j = 0
    for k in e:
        np.savetxt('camera/' + ts + '/imgpts/imgpts' + str(i) + '-' + str(j) + '.txt', k)
        j += 1
    i += 1

# write matrix and distortion coefficients
np.savetxt('camera/' + ts + '/mtx.txt', mtx)
np.savetxt('camera/' + ts + '/dist.txt', dist)

# write rotation and translation vectors
i = 0
for e in rvecs:
    np.savetxt('camera/' + ts + '/rvecs/rvecs' + str(i) + '.txt', e)
    i += 1

i = 0
for e in tvecs:
    np.savetxt('camera/' + ts + '/tvecs/tvecs' + str(i) + '.txt', e)
    i += 1

# write object points
np.savetxt('camera/' + ts + '/objpoints.txt', objpoints[0])

print("wrote camera calibration to camera/" + ts)


# img = cv.imread('images/found/calib_right_138.jpg')
# h,  w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
#
# ###### cv2.remap
# mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
# dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
#
# cv.imwrite('camera/' + ts + '/unwarpresult.png', dst)
# print("wrote unwarped img")
#
# mean_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
#     mean_error += error
# print( "total error: {}".format(mean_error/len(objpoints)) )
