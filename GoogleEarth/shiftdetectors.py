import cv2
import numpy as np
import time

import torch
from lightglue import LightGlue, SuperPoint  # For feature extraction
from lightglue.utils import rbd  # Utility function for removing batch dimension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Light_Matcher = LightGlue(  # Initialize LightGlue. This is used for estimating rotations. 
    features='superpoint',  # Use SuperPoint features
    depth_confidence=-1, #553 with extra 01 # 0.95, -1 IS DEF
    width_confidence=-1, #0.99
    filter_threshold=0.0005  # Custom filter threshold. A lower threshold definitely implies more matches, ie is less accurate / more leniant. The correlation matching is worse with less accuracy. 0.0045. 
).eval().to(device)

def Light_Solver(featsA, featsB):
    """Match features using LightGlue."""
    matches = Light_Matcher({'image0': featsA, 'image1': featsB})
    featsA, featsB, matches = [rbd(x) for x in [featsA, featsB, matches]]
    return featsA, featsB, matches['matches']


# Function to decompose the affine transformation matrix and return the translation
def get_shifts(featsA=None, featsB=None, kp1=None, kp2=None, des1=None, des2=None):
    
    
    ### xxx - pass in kp and none descriptors for alg method
    
    if featsA is None and featsB is None: # ie we are not using the neural network



        if des1 is None or des2 is None:
            raise ValueError("One or both descriptors are None.")

        # Match keypoints using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.999 * n.distance:
                good_matches.append(m)

        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # shifts = dst_pts - src_pts
        # shift_x = np.mean(shifts[0])
        # shift_y = np.mean(shifts[1])
        shift_x, shift_y = get_src_shifts(src_pts, dst_pts)
        print(f"RATREM: {shift_y}")
        return shift_x, shift_y


        center_src = np.mean(src_pts, axis=0).reshape(2, 1)  # Ensure it's a column vector
        center_dst = np.mean(dst_pts, axis=0).reshape(2, 1)  # Ensure it's a column vector

        # Center the points
        src_pts_centered = src_pts - center_src.T  # Transpose to match dimensions
        dst_pts_centered = dst_pts - center_dst.T  # Transpose to match dimensions

        # Calculate the covariance matrix
        H = np.dot(src_pts_centered.reshape(-1, 2).T, dst_pts_centered.reshape(-1, 2))

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)  # Rotation matrix

        # Ensure a right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Translation
        t = center_dst - np.dot(R, center_src)  # This will now work correctly

        # Debugging output
        # print(f"Shape of translation vector t: {t.shape}")

        # Ensure translation has two components
        if t.shape[0] != 2 or t.shape[1] != 1:
            raise ValueError("Translation vector does not have two components.")

        # Convert rotation matrix to angle
        # rotation_angle = np.arctan2(R[1, 0], R[0, 0]) * (180 / np.pi)  # Convert to degrees



    elif featsA is not None and featsB is not None: # We are using the neural network
        # detect and match points with SuperPoint and LightGlue
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
        shifts = dst_pts - src_pts
        shift_x = np.mean(shifts[0])
        shift_y = np.mean(shifts[1])
        return shift_x, shift_y

        center_src = np.mean(src_pts, axis=0).reshape(2, 1)  # Ensure it's a column vector
        center_dst = np.mean(dst_pts, axis=0).reshape(2, 1)  # Ensure it's a column vector

        # Center the points
        src_pts_centered = src_pts - center_src.T  # Transpose to match dimensions
        dst_pts_centered = dst_pts - center_dst.T  # Transpose to match dimensions

        # Calculate the covariance matrix
        H = np.dot(src_pts_centered.reshape(-1, 2).T, dst_pts_centered.reshape(-1, 2))

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)  # Rotation matrix

        # Ensure a right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Translation
        t = center_dst - np.dot(R, center_src)  # This will now work correctly

        # Debugging output
        # print(f"Shape of translation vector t: {t.shape}")

        # Ensure translation has two components
        if t.shape[0] != 2 or t.shape[1] != 1:
            raise ValueError("Translation vector does not have two components.")

        # Convert rotation matrix to angle
        # rotation_angle = np.arctan2(R[1, 0], R[0, 0]) * (180 / np.pi)  # Convert to degrees

    
    return (t[0, 0], t[1, 0])

def get_src_shifts(src_pts, dst_pts):
        shifts = dst_pts - src_pts
        shift_x = np.mean(shifts[0])
        shift_y = np.mean(shifts[1])
        

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

        # Apply the inverse rotation matrix to remove rotation from dst_pts
        dst_pts_rot_corrected = np.dot(dst_pts - center_dst, R.T) + center_dst

        # Now, calculate the translation after removing rotation
        translation = np.mean(dst_pts_rot_corrected - src_pts, axis=0) 
        shift_x = translation[0]
        shift_y = translation[1]
        return shift_x, shift_y

        
        center_src = np.mean(src_pts, axis=0).reshape(2, 1)  # Ensure it's a column vector
        center_dst = np.mean(dst_pts, axis=0).reshape(2, 1)  # Ensure it's a column vector

        # Center the points
        src_pts_centered = src_pts - center_src.T  # Transpose to match dimensions
        dst_pts_centered = dst_pts - center_dst.T  # Transpose to match dimensions

        # Calculate the covariance matrix
        H = np.dot(src_pts_centered.reshape(-1, 2).T, dst_pts_centered.reshape(-1, 2))

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)  # Rotation matrix

        # Ensure a right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Translation
        t = center_dst - np.dot(R, center_src)  # This will now work correctly

        # Debugging output
        # print(f"Shape of translation vector t: {t.shape}")

        # Ensure translation has two components
        if t.shape[0] != 2 or t.shape[1] != 1:
            raise ValueError("Translation vector does not have two components.")

        # Convert rotation matrix to angle
        # rotation_angle = np.arctan2(R[1, 0], R[0, 0]) * (180 / np.pi)  # Convert to degrees
        return (t[0, 0], t[1, 0])  # Access the elements



# DEAD CODE 
'''
        M, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=0.15)
        if np.sum(inliers) < 200:
            M, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=25.0)
            if np.sum(inliers) < 100:
                print("LOW INLIERS")
            
        if M is None:
            print("Affine transformation estimation failed.")
            return None
    

        # Return the translation (xc, yc)
        xc, yc = M[0, 2], M[1, 2]
    else:
        raise ValueError("Invalid input arguments.")

'''