from assign_2_utils import cv_imshow
import cv2
import numpy as np
import time

class my_BFMatcher():

    @staticmethod
    def match(des1, des2):
        match_list = []
        for i, des1_i in enumerate(des1):
            hd_list = np.array([np.count_nonzero((np.unpackbits(des1_i, axis=0) != np.unpackbits(des2_i, axis=0))) for des2_i in des2])

            sorted_hd_list = sorted(hd_list)
            if (sorted_hd_list[0] / sorted_hd_list[1]) >= 0.8:
                continue
            min_i = np.where(hd_list == min(hd_list))[0][0]

            Dmatch = cv2.DMatch()
            Dmatch.queryIdx, Dmatch.trainIdx, Dmatch.distance = i, min_i, float(hd_list[min_i])
            match_list.append(Dmatch)

        return sorted(match_list, key=lambda k : k.distance)

def get_point(kp1, kp2, matches, n=15):
    destP = np.array([kp1[matches[i].queryIdx].pt for i in range(n)])
    srcP = np.array([kp2[matches[i].trainIdx].pt for i in range(n)])
    return srcP, destP

def normalize_feature_points(P):
    mean_P = np.mean(P, axis=0)
    subtracted = P - mean_P
    subtracted_M = [
        [1, 0, -mean_P[0]],
        [0, 1, -mean_P[1]],
        [0, 0, 1]
    ]
    
    scaled_P = np.sqrt(2) / np.hypot(subtracted[:,0], subtracted[:,1]).max()
    scaled_M = [
        [scaled_P, 0, 0],
        [0, scaled_P, 0],
        [0, 0, 1]
    ]
    
    T = np.dot(scaled_M, subtracted_M)
    return T

def Apply_T(T, P):
    homo_coor = np.concatenate((P, np.ones((P.shape[0], 1))), axis=1)
    
    applied = []
    for i in homo_coor:
        tmp = T @ i.reshape(-1,1)
        applied.append(tmp / tmp[2])
    
    return np.array(applied)[:, :2][:,:,0]

def compute_homography(srcP, destP):

    Ts, Td = normalize_feature_points(srcP), normalize_feature_points(destP)
    
    normalized_s, normalized_d = Apply_T(Ts, srcP), Apply_T(Td, destP)


    size = normalized_s.shape[0]
    A = np.zeros((2*size, 9))
    for i in range(size):
        x, y = normalized_s[i,0], normalized_s[i,1]
        x_prime, y_prime = normalized_d[i,0], normalized_d[i,1]
        A[i*2] = [-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime]
        A[i*2+1] = [0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime]

    _, _, V = np.linalg.svd(A)

    H = V[-1].reshape([3,3])
    return H / H[2,2], Ts, Td

def Image_wraping(base, output):
    return np.where(output == 0, base, output)

def compute_homography_ransac(srcP, destP, th):
    '''
    Loop:
        1. Randomly select a four point correspondences
        2. Compute H
        3. Count inliers to the current H
        4. Keep H if largest number of inliers
        (until 3 seconds)
    5. Recompute H using all inliers
    '''
    np.seterr(divide='ignore', invalid='ignore')
    start = time.time()
    best_cnt = -1
    best_inlier = None
    time_limit = 3
    while time.time() - start < time_limit:
        random_idx = np.random.choice(len(srcP), 4, replace=False)

        H, Ts, Td = compute_homography(np.array(srcP[random_idx]), np.array(destP[random_idx]))
        output_H = ((np.linalg.inv(Td)@H)@Ts)
        my_destP = Apply_T(output_H, srcP)

        reproj = np.hypot((my_destP - destP)[:,0], (my_destP - destP)[:,1])
        inlier = reproj < th
        inlier_idx = np.where(inlier == True)[0]
        inlier_cnt = len(inlier_idx)

        if best_cnt < inlier_cnt:
            # 제일 좋은거
            best_cnt, best_inlier = inlier_cnt, inlier_idx
    
    return compute_homography(srcP[best_inlier], destP[best_inlier])

def blending(left, right, combined, scope):
    boundary = left.shape[1]
    for i in range(scope):
        a = i / scope
        combined[:,boundary-scope+i] = (1-a)*left[:,i-scope] + a*right[:, boundary-scope+i]
    return combined

if __name__ == '__main__':

    # 2-1)
    img_desk = cv2.imread('Assign2/cv_desk.png', cv2.IMREAD_GRAYSCALE)
    img_cover = cv2.imread('Assign2/cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
    if img_desk is None or img_cover is None:
        print('Image load failed!')
        exit()
    
    orb = cv2.ORB_create()

    kp_desk = orb.detect(img_desk, None)
    kp_desk, des_desk = orb.compute(img_desk, kp_desk)

    kp_cover = orb.detect(img_cover, None)
    kp_cover, des_cover = orb.compute(img_cover, kp_cover)

    matches = my_BFMatcher.match(des_desk, des_cover)
    result = cv2.drawMatches(img_desk, kp_desk, img_cover, kp_cover, matches[:10], None, flags=2)
    cv_imshow('feature matching', result)

    # 2-2)
    srcP, destP = get_point(kp1 = kp_desk, kp2 = kp_cover, matches=matches, n=15)

    H, Ts, Td = compute_homography(srcP, destP)
    H_n = ((np.linalg.inv(Td)@H)@Ts)
    output_n = cv2.warpPerspective(img_cover, H_n, (img_desk.shape[1], img_desk.shape[0]))
    cv_imshow('Homography', output_n)
    cv_imshow('Homography wraping', Image_wraping(img_desk, output_n))

    # 2-3)
    H, Ts, Td = compute_homography_ransac(srcP, destP, th=4)
    H_r = ((np.linalg.inv(Td)@H)@Ts)
    output_r = cv2.warpPerspective(img_cover, H_r, (img_desk.shape[1], img_desk.shape[0]))
    cv_imshow('Homography with RANSAC', output_r)
    cv_imshow('Homography with RANSAC wraping', Image_wraping(img_desk, output_r))

    # 2-4-c)
    img_hp = cv2.imread('Assign2/hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
    if img_hp is None:
        print('Image load failed!')
        exit()

    img_hp = cv2.resize(img_hp, (img_cover.shape[1], img_cover.shape[0]))
    output_hp = cv2.warpPerspective(img_hp, H_r, (img_desk.shape[1], img_desk.shape[0]))
    cv_imshow('Homography with RANSAC (Harry Potter)', output_hp)
    cv_imshow('Homography with RANSAC wraping (Harry Potter)', Image_wraping(img_desk, output_hp))

    # 2-5 -a)
    img_d10 = cv2.imread('Assign2/diamondhead-10.png', cv2.IMREAD_GRAYSCALE)
    img_d11 = cv2.imread('Assign2/diamondhead-11.png', cv2.IMREAD_GRAYSCALE)
    if img_d10 is None or img_d11 is None:
        print('Image load failed!')
        exit()
    
    orb = cv2.ORB_create()

    kp_d10 = orb.detect(img_d10, None)
    kp_d10, des_d10 = orb.compute(img_d10, kp_d10)

    kp_d11 = orb.detect(img_d11, None)
    kp_d11, des_d11 = orb.compute(img_d11, kp_d11)

    matches = my_BFMatcher.match(des_d10, des_d11)
    srcP, destP = get_point(kp1 = kp_d10, kp2 = kp_d11, matches=matches, n=15)

    H, Ts, Td= compute_homography_ransac(srcP, destP, th=1)
    H_r = ((np.linalg.inv(Td)@H)@Ts)

    combined = cv2.warpPerspective(img_d11, H_r, ((img_d11.shape[1] + img_d10.shape[1]), img_d10.shape[0]))
    right = combined.copy()
    combined[:img_d10.shape[0], :img_d10.shape[1]] = img_d10
    cv_imshow('Image stitching',combined)

    # 2-5 -b)
    cv_imshow('Image stitching and blending',blending(img_d10, right, combined, 100))
