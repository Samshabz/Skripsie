
import numpy as np


# Affine Transformation Estimation (Using OpenCV)
import cv2

# Input: src_pts and dst_pts should be numpy arrays with shape (N, 2)
def affine_transformation(src_pts, dst_pts):
    ransac_threshold = 25
    matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold)

    # Translation components
    tx = matrix[0, 2]
    ty = matrix[1, 2]
    
    # Rotation component (from the affine matrix)
    rotation_angle = np.arctan2(matrix[1, 0], matrix[0, 0])

    return tx, ty, np.degrees(rotation_angle)


# Rigid Transformation Estimation (Using SVD)
from scipy.linalg import svd

# Input: src_pts and dst_pts should be numpy arrays with shape (N, 2)
def rigid_transformation(src_pts, dst_pts):
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    U, S, Vt = svd(np.dot(dst_centered.T, src_centered))
    R = np.dot(U, Vt)
    t = dst_mean - np.dot(src_mean, R)
    
    # Translation components
    tx, ty = t
    
    # Rotation component
    rotation_angle = np.arctan2(R[1, 0], R[0, 0])

    return tx, ty, np.degrees(rotation_angle)


# Homography Estimation (Using OpenCV)
def homography_transformation(src_pts, dst_pts):
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=15)
    
    # Translation components from the homography matrix
    tx = H[0, 2]
    ty = H[1, 2]
    
    # Rotation component
    rotation_angle = np.arctan2(H[1, 0], H[0, 0])

    return tx, ty, np.degrees(rotation_angle)


    
# RANSAC for Affine Transformation (Using OpenCV)
def ransac_affine_transformation(src_pts, dst_pts):
    matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    
    # Translation components
    tx = matrix[0, 2]
    ty = matrix[1, 2]
    
    # Rotation component
    rotation_angle = np.arctan2(matrix[1, 0], matrix[0, 0])

    return tx, ty, np.degrees(rotation_angle)
