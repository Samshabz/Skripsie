import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
import time

import torch
from lightglue import LightGlue, SuperPoint  # For feature extraction
from lightglue.utils import rbd  # Utility function for removing batch dimension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Light_Matcher = LightGlue(  # Initialize LightGlue. This is used for estimating rotations. 
    features='superpoint',  # Use SuperPoint features
    # depth_confidence=0.95, #553 with extra 01 # 0.95, -1 IS DEF
    # width_confidence=0.99, #0.99
    filter_threshold=0.0025,  # Custom filter threshold. A lower threshold definitely implies more matches, ie is less accurate / more leniant. The correlation matching is worse with less accuracy. 0.0045. 
    n_layers = 6,  # Reduce layers for faster inference
    flash = False,  # FlashAttention remains enabled for speed. check if it needs CUDA
    mp = False,  # Enable mixed precision for faster inference. check if it needs CUDA
    depth_confidence = 0.9,  # Stop earlier to speed up
    width_confidence = 0.95  # Prune points earlier for efficiency
    # filter_threshold = 0.2  # Increase match confidence for fewer but more robust matches

).eval().to(device)

def Light_Solver(featsA, featsB):
    """Match features using LightGlue."""
    matches = Light_Matcher({'image0': featsA, 'image1': featsB})
    featsA, featsB, matches = [rbd(x) for x in [featsA, featsB, matches]]
    return featsA, featsB, matches['matches']


# Function to decompose the affine transformation matrix and return the translation
def get_neural_src_pts(featsA=None, featsB=None, kp1=None, kp2=None, des1=None, des2=None):
    

    if len(featsA['keypoints'][0]) < 10 or len(featsB['keypoints'][0]) < 10:

        raise ValueError("Not enough keypoints for matching.")
        return None
    featsA, featsB, matches = Light_Solver(featsA, featsB)


    good_matches = matches[:]

    if len(good_matches) < 4:
        print("Warning: Less than 4 matches found.")
        return None, None, None
        
    keypoints1 = featsA['keypoints'].cpu().numpy() # the .cpu() is used to move the tensor to the cpu. The .numpy() is used to convert the tensor to a numpy array
    keypoints2 = featsB['keypoints'].cpu().numpy()
    matches = matches.cpu().numpy()

    src_pts = keypoints1[matches[:, 0]]
    dst_pts = keypoints2[matches[:, 1]]
    # shifts = dst_pts - src_pts
    return src_pts, dst_pts, good_matches


def get_src_shifts(src_pts, dst_pts, ret_angle=False):
        # this code improves response by applying a secondary rotational normalization step, in theory it should have minimal impact. 

        center_src = np.mean(src_pts, axis=0)
        center_dst = np.mean(dst_pts, axis=0)

        # Center the points around their mean
        src_pts = src_pts.reshape(-1, 2)
        dst_pts = dst_pts.reshape(-1, 2)
        src_pts_centered = src_pts - center_src
        dst_pts_centered = dst_pts - center_dst

        # Estimate the rotation matrix using Singular Value Decomposition (SVD)
        H = np.dot(src_pts_centered.T, dst_pts_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(U, Vt)

        theta = np.arctan2(R[1, 0], R[0, 0])
        theta_deg = np.degrees(theta)
        if ret_angle:
            return theta_deg
        # print(f"Rotation angle (degrees): {theta_deg}")

        # Apply the inverse rotation matrix to remove rotation from dst_pts
        dst_pts_rot_corrected = np.dot(dst_pts - center_dst, R.T) + center_dst

        # Now, calculate the translation after removing rotation
        translation = np.mean(dst_pts_rot_corrected - src_pts, axis=0) 
        shift_x = translation[0]
        shift_y = translation[1]
        return shift_x, shift_y
