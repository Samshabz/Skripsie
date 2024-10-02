# Procrustes Analysis (Using scipy)
from scipy.spatial import procrustes
import numpy as np

# Input: src_pts and dst_pts should be numpy arrays with the same shape (N, 2)
def procrustes_analysis(src_pts, dst_pts):
    mtx1, mtx2, disparity = procrustes(src_pts, dst_pts)
    
    # Compute translation as the difference in centroids
    tx = np.mean(mtx2[:, 0]) - np.mean(mtx1[:, 0])
    ty = np.mean(mtx2[:, 1]) - np.mean(mtx1[:, 1])
    
    # Compute rotation (since Procrustes can involve scaling and rotation)
    rotation_matrix = np.linalg.lstsq(mtx1, mtx2, rcond=None)[0][:2, :2]
    rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    
    return tx, ty, np.degrees(rotation_angle)


# ICP (Using open3d)
import open3d as o3d

def icp_2d(src_pts, dst_pts, threshold=0.02):
    src_pts = np.array(src_pts).reshape(-1, 2)
    dst_pts = np.array(dst_pts).reshape(-1, 2)
    
    def create_point_cloud(pts):
        pts_3d = np.hstack([pts, np.zeros((pts.shape[0], 1))])
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts_3d)
        return pc

    src_cloud = create_point_cloud(src_pts)
    dst_cloud = create_point_cloud(dst_pts)

    trans_init = np.eye(4)
    icp_result = o3d.pipelines.registration.registration_icp(
        src_cloud, dst_cloud, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    transformation = icp_result.transformation
    tx = transformation[0, 3]
    ty = transformation[1, 3]
    rotation_matrix = transformation[:2, :2]
    rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return tx, ty, np.degrees(rotation_angle)


# Affine Transformation Estimation (Using OpenCV)
import cv2

# Input: src_pts and dst_pts should be numpy arrays with shape (N, 2)
def affine_transformation(src_pts, dst_pts):
    matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts)

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
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    
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
