# You should replace these 3 lines with the output in calibration step
# DIM=XXX
# K=np.array(YYY)
# D=np.array(ZZZ)
import numpy as np
import cv2
import sys

DIM = (3072, 2048)
K = np.array([[1511.7515811351727, 0.0, 1496.5712039727023], [0.0, 1497.8201412252645, 1034.2071726674196], [0.0, 0.0, 1.0]])
D = np.array([[-0.03820374941567194], [-0.010632408359622545], [0.010231867812191526], [-0.0034649456540657756]])

def undistort(img_path, balance=1.0, dim2=None, dim3=None):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        # dim3 = dim1
        scale = 2.0
        dim3 = (int(dim1[0]*scale), int(dim1[1]*scale))
    # scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    print(new_K)
    new_K[0][2] = 3000.0
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite('undistorted.jpg', undistorted_img)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort(p)



# def undistort(img_path):
#     img = cv2.imread(img_path)
#     h,w = img.shape[:2]
#     map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
#     undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#     cv2.imwrite('undistorted.jpg', undistorted_img)
#     cv2.imshow("undistorted", undistorted_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# if __name__ == '__main__':
#     for p in sys.argv[1:]:
#         undistort(p)
