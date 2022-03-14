import cv2
import numpy as np
import matplotlib.pyplot as plt


def rectify(image, K, D, balance=1.0, scale=1.0, dim2=None, dim3=None):
    """
    rectify wide angle distortion
    credit: https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
    """
    assert K[2][2] == 1.0
    dim1 = image.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = (int(dim1[0]*scale), int(dim1[1]*scale))
    
    # scaled_K, dim2 and balance are used to determine the final K used to un-distort image. 
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim2, np.eye(3), balance=balance)

    ## scale translation vectors accordingly
    new_K[0][2] *= scale
    new_K[1][2] *= scale
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    rectified_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rectified_img, new_K
    

def getSIFTMatchesPair(img1, img2, maskL, maskR):
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
    ## init sift object
    sift = cv2.SIFT_create()
    corners1, corners2, descriptors1, descriptors2 = [], [], [], []
    
    # find image keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,mask=maskL) ## 'left' image
    kp2, des2 = sift.detectAndCompute(img2,mask=maskR) ## 'right' image
    
    # use BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    for m,n in matches:
        if m.distance < 0.75 * n.distance: ## filter poor matches
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
    """
    plot SIFT features with the image
    """
    feat_x, feat_y = [], []
    for i in range(len(features)):
        feat_x.append(features[i][0])
        feat_y.append(features[i][1])
    fig, ax = plt.subplots(figsize=(12,6))
    ax.imshow(img, cmap='gray')
    ax.scatter(feat_x, feat_y)
    plt.show()
    
    
def plotFeatureCorr(img1, img2, c1, c2, mask):
    """
    plot SIFT feature matches in two images
    """
    c1_y, c1_x, c2_y, c2_x = [], [], [], []
    assert len(c1) == len(c2)
    for i in range(len(c1)):
        if mask[i] == 1:
            c1_x.append(c1[i][0])
            c1_y.append(c1[i][1])
            c2_x.append(c2[i][0])
            c2_y.append(c2[i][1])
    fig, ax = plt.subplots(1,2, figsize=(20,6))
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(img2, cmap='gray')
    ax[0].scatter(c1_x, c1_y)
    ax[1].scatter(c2_x, c2_y)
    plt.show()
    
    
def cylindricalWarp(img, K):
    """
    Fast cylindical warp.
    returns the cylindrical warp for a given image and intrinsics matrix K.
    credit: https://www.morethantechnical.com/2018/10/30/cylindrical-image-warping-for-panorama-stitching/
    """
    h_,w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_,w_))
    X = np.stack([x_i,y_i,np.ones_like(x_i)],axis=-1).reshape(h_*w_,3) # to homog
    Kinv = np.linalg.inv(K) 
    X = Kinv.dot(X.T).T # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:,0]),X[:,1],np.cos(X[:,0])],axis=-1).reshape(w_*h_,3)
    B = K.dot(A.T).T # project back to image-pixels plane
    # back from homog coords
    B = B[:,:-1] / B[:,[-1]]
    # make sure warp coords only within image bounds
    B[(B[:,0] < 0) | (B[:,0] >= w_) | (B[:,1] < 0) | (B[:,1] >= h_)] = -1
    B = B.reshape(h_,w_,-1)
    
    img_rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:,:,0].astype(np.float32), B[:,:,1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

