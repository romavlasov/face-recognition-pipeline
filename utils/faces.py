import cv2
import numpy as np


def face_align(img, landmarks, d_size=(400, 400), normalized=False, show=False):
    ref_landmarks = np.array([30.2946 / 96, 51.6963 / 112,
                              65.5318 / 96, 51.5014 / 112,
                              48.0252 / 96, 71.7366 / 112,
                              33.5493 / 96, 92.3655 / 112,
                              62.7299 / 96, 92.2041 / 112], dtype=np.float64).reshape(5, 2)
        
    landmarks = np.array(landmarks).reshape(5, 2)
    dw, dh = d_size

    keypoints = landmarks.copy().astype(np.float64)
    if normalized:
        keypoints[:, 0] *= img.shape[1]
        keypoints[:, 1] *= img.shape[0]

    keypoints_ref = np.zeros((5, 2), dtype=np.float64)
    keypoints_ref[:, 0] = ref_landmarks[:, 0] * dw
    keypoints_ref[:, 1] = ref_landmarks[:, 1] * dh

    transform_matrix = transformation_from_points(keypoints_ref, keypoints)
    output_im = cv2.warpAffine(img, transform_matrix, d_size, flags=cv2.WARP_INVERSE_MAP)

    return output_im


def transformation_from_points(points1, points2):
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    u, _, vt = np.linalg.svd(np.matmul(points1.T, points2))
    r = np.matmul(u, vt).T

    return np.hstack(((s2 / s1) * r, (c2.T - (s2 / s1) * np.matmul(r, c1.T)).reshape(2, -1)))
