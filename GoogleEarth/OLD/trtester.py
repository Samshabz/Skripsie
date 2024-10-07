import cv2
import numpy as np
from scipy.spatial import procrustes
import open3d as o3d
from scipy.linalg import svd

# Function to crop top and bottom 10% of the image
def crop_image(img):
    height, width = img.shape[:2]
    crop_size = int(height * 0.15)  # 10% of the height
    return img[crop_size:height-crop_size, :]  # Crop top and bottom

def extract_keypoints(image1_gray, image2_gray):
    """Detect and compute keypoints and descriptors using AKAZE"""
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(image1_gray, None)
    kp2, des2 = akaze.detectAndCompute(image2_gray, None)

    if des1 is None or des2 is None:
        raise ValueError("Could not find keypoints or descriptors in one or both images.")

    # BFMatcher for matching keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    return src_pts, dst_pts


# Procrustes Analysis
def procrustes_analysis(src_pts, dst_pts):
    mtx1, mtx2, disparity = procrustes(src_pts, dst_pts)
    tx = np.mean(mtx2[:, 0]) - np.mean(mtx1[:, 0])
    ty = np.mean(mtx2[:, 1]) - np.mean(mtx1[:, 1])
    rotation_matrix = np.linalg.lstsq(mtx1, mtx2, rcond=None)[0][:2, :2]
    rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return tx, ty, np.degrees(rotation_angle)


# ICP (Using open3d)
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
def affine_transformation(src_pts, dst_pts):
    matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts)
    tx = matrix[0, 2]
    ty = matrix[1, 2]
    rotation_angle = np.arctan2(matrix[1, 0], matrix[0, 0])
    return tx, ty, np.degrees(rotation_angle)


# Rigid Transformation Estimation (Using SVD)
def rigid_transformation(src_pts, dst_pts):
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    U, S, Vt = svd(np.dot(dst_centered.T, src_centered))
    R = np.dot(U, Vt)
    t = dst_mean - np.dot(src_mean, R)
    
    tx, ty = t
    rotation_angle = np.arctan2(R[1, 0], R[0, 0])
    return tx, ty, np.degrees(rotation_angle)


# Homography Estimation (Using OpenCV)
def homography_transformation(src_pts, dst_pts):
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    tx = H[0, 2]
    ty = H[1, 2]
    rotation_angle = np.arctan2(H[1, 0], H[0, 0])
    return tx, ty, np.degrees(rotation_angle)

def phase_correlation_translation(image1, image2):
    """Estimate translation using Phase Correlation."""
    shift, response = cv2.phaseCorrelate(np.float32(image1), np.float32(image2))
    translation_x, translation_y = shift
    return translation_x, translation_y


# RANSAC for Affine Transformation (Using OpenCV)
def ransac_affine_transformation(src_pts, dst_pts):
    matrix, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    tx = matrix[0, 2]
    ty = matrix[1, 2]
    rotation_angle = np.arctan2(matrix[1, 0], matrix[0, 0])
    return tx, ty, np.degrees(rotation_angle)


# Main function to run the analysis
def main():
    # Load two images
    image1_path = './GoogleEarth/DATASETS/BASIC/1.jpg'
    image2_path = './GoogleEarth/DATASETS/BASIC/3.jpg'

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Error: Could not load one or both images.")
        return

    image1_cropped = crop_image(image1)
    image2_cropped = crop_image(image2)

    image1_gray = cv2.cvtColor(image1_cropped, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2_cropped, cv2.COLOR_BGR2GRAY)

    # Extract keypoints and descriptors
    src_pts, dst_pts = extract_keypoints(image1_gray, image2_gray)

    # Apply different transformation estimation methods
    print("Comparing methods:")
    tx_truth, ty_truth = phase_correlation_translation(image1_gray, image2_gray)
    print(f"Phase Correlation: Translation (x, y): ({tx_truth}, {ty_truth})")

    # tx_p, ty_p, rot_p = procrustes_analysis(src_pts, dst_pts)
    # print(f"Procrustes: Translation (x, y): ({tx_p}, {ty_p}), Rotation: {rot_p}°")

    # tx_icp, ty_icp, rot_icp = icp_2d(src_pts, dst_pts)
    # print(f"ICP: Translation (x, y): ({tx_icp}, {ty_icp}), Rotation: {rot_icp}°")

    tx_aff, ty_aff, rot_aff = affine_transformation(src_pts, dst_pts)
    print(f"Affine: Translation (x, y): ({tx_aff}, {ty_aff}), Rotation: {rot_aff}°")

    tx_rigid, ty_rigid, rot_rigid = rigid_transformation(src_pts, dst_pts)
    print(f"Rigid: Translation (x, y): ({tx_rigid}, {ty_rigid}), Rotation: {rot_rigid}°")

    tx_hom, ty_hom, rot_hom = homography_transformation(src_pts, dst_pts)
    print(f"Homography: Translation (x, y): ({tx_hom}, {ty_hom}), Rotation: {rot_hom}°")

    tx_ransac, ty_ransac, rot_ransac = ransac_affine_transformation(src_pts, dst_pts)
    print(f"RANSAC Affine: Translation (x, y): ({tx_ransac}, {ty_ransac}), Rotation: {rot_ransac}°")


if __name__ == "__main__":
    main()
