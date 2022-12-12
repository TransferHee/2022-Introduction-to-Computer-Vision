import cv2
import numpy as np
import time
from compute_avg_reproj_error import compute_avg_reproj_error
'''
2018314848 한승희
Introduction to Computer Vision
Assignment #3
env:
    Python 3.7.9
    numpy 1.12.6
    opencv-python(cv2) 4.6.0
파일 경로 불러올때 모든 파일이 Assign3/{img_name}.png 와 같은 형식으로 되어있습니다
체점하실때 이 부분을 잘 변경해서 해주시면 감사하겠습니다!
'''

def compute_F_raw(M):
    A = [[x*xp, xp*y, xp, x*yp, y*yp, yp, x, y, 1] for x, y, xp, yp in M]
    _, _, V = np.linalg.svd(A)
    return V[-1].reshape(3, 3)

def normalize_points(img_shape):
    h, w = img_shape
    subtracted_M = [
        [1, 0, -w/2],
        [0, 1, -h/2],
        [0, 0, 1]
    ]
    scaled_M = [
        [2/w, 0, 0],
        [0, 2/h, 0],
        [0, 0, 1]
    ]
    T = np.dot(scaled_M, subtracted_M)
    return T

def Apply_T(T, P):
    applied = []
    for x, y, xp, yp in P:
        new_x, new_y, s_1 = T @ np.array([x, y, 1])
        new_xp, new_yp, s_2 = T @ np.array([xp, yp, 1])
        applied.append([new_x/s_1, new_y/s_1, new_xp/s_2, new_yp/s_2])
    return np.array(applied)

def compute_F_norm(M):
    # Normalize points
    global img_shape
    T = normalize_points(img_shape)
    normalized_M = Apply_T(T, M)

    # Compute F
    F = compute_F_raw(normalized_M)
    
    # Denormalize F
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V

    return T.T @ F @ T
    
def compute_F_mine(M):
    # ransac 알고리즘을 활용한 것인데, assign2에서 랜삭 포인트를 4개로 고정한 것과 달리 
    # 피처 매칭을 몇개를 둘 건지도 랜덤으로(40~70개) 골라서 그 중 최고의 F를 찾는 방식
    global img_shape
    T = normalize_points(img_shape)

    best_error = 10
    best_F = None
    start = time.time()
    while time.time() - start < 2.99:
        feature_num = np.random.choice(range(40, 70))
        random_idx = np.random.choice(len(M), feature_num, replace=False)
        normalized_M = Apply_T(T, M[random_idx])

        F = compute_F_raw(normalized_M)

        U, S, V = np.linalg.svd(F)
        S[-1] = 0
        F = U @ np.diag(S) @ V
        current_F = T.T @ F @ T
        
        error = compute_avg_reproj_error(M, current_F)
        if error < best_error:
            best_error = error
            best_F = current_F
    # print(feature_num)
    return best_F

def compute_epipolar_lines(F, M):
    lines_1, lines_2 = [], []
    for x, y, xp, yp in M:
        lines_1.append(F @ np.array([x, y, 1]))
        lines_2.append(F.T @ np.array([xp, yp, 1]))
    return np.array(lines_1), np.array(lines_2)

def drawlines(img, lines, matches):
    _, h, _ = img.shape
    colors = [(255,0,0), (0,255,0), (0,0,255)]
    output = img.copy()
    for i in range(len(matches)):
        x0, y0 = map(int, [0, -lines[i][2]/lines[i][1]])
        x1, y1 = map(int, [h, -(lines[i][2]+lines[i][0]*h)/lines[i][1]])
        output = cv2.line(output, (x0, y0), (x1, y1), colors[i], 2)
        output = cv2.circle(output, tuple(matches[i, :2].astype(np.int32)), 5, colors[i], -1)
    return output

def Fundamental_matrix_computation(img1, img1_name, img2_name, M):
    global img_shape
    img_shape = img1.shape[:2]
    print(f'Average Reprojection Errors ({img1_name} and {img2_name})')
    print('Raw =',compute_avg_reproj_error(M, compute_F_raw(M)))
    print('Norm =',compute_avg_reproj_error(M, compute_F_norm(M)))
    F_mine = compute_F_mine(M)
    print('Mine =',compute_avg_reproj_error(M, F_mine))
    print('OpenCV 8POINT =',compute_avg_reproj_error(M, cv2.findFundamentalMat(M[:, :2], M[:, 2:], cv2.FM_8POINT)[0]))
    print()
    return F_mine

if __name__ == '__main__':
    # temples
    t1, t2 = cv2.imread('Assign3/temple1.png'), cv2.imread('Assign3/temple2.png')
    if t1 is None or t2 is None:
        print("Cannot read image")
        exit()
    t1_name, t2_name = 'temple1.png', 'temple2.png'
    M = np.loadtxt('Assign3/temple_matches.txt')
    F_t = Fundamental_matrix_computation(t1, t1_name, t2_name, M)

    # house
    h1, h2 = cv2.imread('Assign3/house1.jpg'), cv2.imread('Assign3/house2.jpg')
    if h1 is None or h2 is None:
        print("Cannot read image")
        exit()
    h1_name, h2_name = 'house1.jpg', 'house2.jpg'
    M = np.loadtxt('Assign3/house_matches.txt')
    F_h = Fundamental_matrix_computation(h1, h1_name, h2_name, M)

    # library
    l1, l2 = cv2.imread('Assign3/library1.jpg'), cv2.imread('Assign3/library2.jpg')
    if l1 is None or l2 is None:
        print("Cannot read image")
        exit()
    l1_name, l2_name = 'library1.jpg', 'library2.jpg'
    M = np.loadtxt('Assign3/library_matches.txt')
    F_l = Fundamental_matrix_computation(l1, l1_name, l2_name, M)

    # Epipolar lines
    # temples
    print('Epipolar lines of temples')
    key = 0
    while key != ord('q'):
        random_idx = np.random.choice(len(M), 3 , replace=False)
        lines_1, lines_2 = compute_epipolar_lines(F_t, M[random_idx])
        output_1 = drawlines(t1, lines_1, M[random_idx])
        output_2 = drawlines(t2, lines_2, M[random_idx])
        output = np.concatenate([output_1, output_2], axis=1)
        cv2.imshow('Epipolar lines of temples', output)
        key = cv2.waitKey()
    cv2.destroyAllWindows()

    # house
    print('Epipolar lines of house')
    key = 0
    while key != ord('q'):
        random_idx = np.random.choice(len(M), 3 , replace=False)
        lines_1, lines_2 = compute_epipolar_lines(F_h, M[random_idx])
        output_1 = drawlines(h1, lines_1, M[random_idx])
        output_2 = drawlines(h2, lines_2, M[random_idx])
        output = np.concatenate([output_1, output_2], axis=1)
        cv2.imshow('Epipolar lines of house', output)
        key = cv2.waitKey()
    cv2.destroyAllWindows()

    # library
    print('Epipolar lines of library')
    key = 0
    while key != ord('q'):
        random_idx = np.random.choice(len(M), 3 , replace=False)
        lines_1, lines_2 = compute_epipolar_lines(F_l, M[random_idx])
        output_1 = drawlines(l1, lines_1, M[random_idx])
        output_2 = drawlines(l2, lines_2, M[random_idx])
        output = np.concatenate([output_1, output_2], axis=1)
        cv2.imshow('Epipolar lines of library', output)
        key = cv2.waitKey()
    cv2.destroyAllWindows()