import numpy as np
import cv2
from skimage import transform as tf

from helpers import *

import scipy

def getFeatures(img,bbox):
    """
    Description: Identify feature points within bounding box for each object
    Input:
        img: Grayscale input image, (H, W)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in first frame, (F, N, 2)
    Instruction: Please feel free to use cv2.goodFeaturesToTrack() or cv.cornerHarris()
    """
    # Set parameters for Shi-Tomasi feature detection
    maxCorners = 25  # Max number of corners
    qualityLevel = 0.01  # the minimum quality of corner below which everyone is rejected
    minDistance = 10  # throws away all the nearby corners in the range of minimum distance

    F = bbox.shape[0]  # number of bounding boxes
    N = maxCorners
    features = -np.ones((F, N, 2))

    for i in range(F):
        x_tl, y_tl, x_br, y_br =  int(bbox[i, 0, 0]), int(bbox[i, 0, 1]), int(bbox[i, 1, 0]), int(bbox[i, 1, 1])
        img_bbox = img[y_tl:y_br+1, x_tl:x_br+1]

        # Shi-Tomasi feature detection
        corners = cv2.goodFeaturesToTrack(img_bbox, maxCorners, qualityLevel, minDistance)
        # corners = np.int0(corners)

        # Current feature points coordinates are in bbox image, convert into original image
        corners[:, 0, 0] += x_tl
        corners[:, 0, 1] += y_tl

        features[i, 0:len(corners), :] = corners[:, 0]

    return features


def estimateFeatureTranslation(feature, Ix, Iy, img1, img2):
    """
    Description: Get corresponding point for one feature point
    Input:
        feature: Coordinate of feature point in first frame, (2,)
        Ix: Gradient along the x direction, (H,W)
        Iy: Gradient along the y direction, (H,W)
        img1: First image frame, (H,W,3)
        img2: Second image frame, (H,W,3)
    Output:
        new_feature: Coordinate of feature point in second frame, (2,)
    Instruction: Please feel free to use interp2() and getWinBound() from helpers
    """

    It = img2 - img1
    A = np.hstack((Ix.reshape(-1, 1), Iy.reshape(-1, 1)))
    b = -It.reshape(-1, 1)
    res = np.linalg.solve(A.T @ A, A.T @ b)

    new_feature = feature + res[:, 0]

    return new_feature


def estimateAllTranslation(features, img1, img2):
    """
    Description: Get corresponding points for all feature points
    Input:
        features: Coordinates of all feature points in first frame, (F, N, 2)
        img1: First image frame, (H,W,3)
        img2: Second image frame, (H,W,3)
    Output:
        new_features: Coordinates of all feature points in second frame, (F, N, 2)
    """
    Jx, Jy = findGradient(img2, 7, 1.3)

    F = features.shape[0]
    N = features.shape[1]

    new_features = -np.ones((F, N, 2))

    for f in range(F):
        for i in range(N):
            feature = features[f, i, :]
            new_feature = estimateFeatureTranslation(feature, Jx, Jy, img1, img2)
            new_features[f, i, :] = new_feature

    return new_features


def findGradient(img, ksize=5, sigma=1):
    G = cv2.getGaussianKernel(ksize, sigma)
    G = G @ G.T
    fx = np.array([[1, -1]])
    fy = fx.T
    Gx = scipy.signal.convolve2d(G, fx, 'same', 'symm')[:, 1:]
    Gy = scipy.signal.convolve2d(G, fy, 'same', 'symm')[1:, :]
    Ix = scipy.signal.convolve2d(img, Gx, 'same', 'symm')
    Iy = scipy.signal.convolve2d(img, Gy, 'same', 'symm')
    return Ix, Iy


def applyGeometricTransformation(features, new_features, bbox):
    """
    Description: Transform bounding box corners onto new image frame
    Input:
        features: Coordinates of all feature points in first frame, (F, N, 2)
        new_features: Coordinates of all feature points in second frame, (F, N, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Output:
        features: Coordinates of all feature points in second frame after eliminating outliers, (F, N1, 2)
        bbox: Top-left and bottom-right corners of all bounding boxes, (F, 2, 2)
    Instruction: Please feel free to use skimage.transform.estimate_transform()
    """

    F = bbox.shape[0]  # number of bounding boxes
    N = features.shape[1] # number of feature points

    new_boxes = -np.ones((F, 2, 2))
    sec_features = -np.ones((F, N, 2))

    for f in range(F):
        tform = tf.estimate_transform('similarity', features[f], new_features[f])
        # print("this is estimate transform ", tform)

        new_box = tform(bbox[f, :, :])
        x_tl, y_tl, x_br, y_br = int(new_box[0, 0]), int(new_box[0, 1]), int(new_box[1, 0]), int(new_box[1, 1])
        # print("This is current box", new_box)

        sec_feature = tform(features[f, :, :])
        # print("This is transformed features", sec_features)

        # Eliminating outliers
        # Need to filter out invalid new_feature, according to new box coordinate
        threshold = 1
        for i in range(len(sec_feature)):

            # 1. If a feature point moves too much (how much? you can tune your own distance threshold),
            # then eliminate this feature because it probably failed in tracking.
            diff = np.linalg.norm((features - sec_feature)[f][i])
            if diff > threshold:
                # print("Distance exceeds threshold.")
                sec_feature[i][0], sec_feature[i][1] == -1, -1

            # 2. If a feature point is outside of the new calculated bbox, then eliminate this feature.
            if (sec_feature[i][0] < x_tl) or (sec_feature[i][0] > x_br) or (sec_feature[i][1] < y_tl) or (sec_feature[i][1] > y_br):
                # print("Out of bound")
                sec_feature[i][0], sec_feature[i][1] == -1, -1


        # valid_features = np.where(new_features[f,:,:])
        # for i in range(N):
        #     new_feature = new_features[f, i, :]
        #     if (new_feature[0] > new_box[0][0] and new_feature[0] > new_box[1][0] and new_feature[1] > new_box[0][1]  and new_feature[1] < new_box[1][1]):

        # print("this is current new_feature", new_features[f,:,:])
        new_boxes[f] = new_box
        sec_features[f] = sec_feature


    return sec_features, new_boxes


