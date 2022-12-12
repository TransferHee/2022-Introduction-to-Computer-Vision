import cv2
import numpy as np
'''
2018314848 한승희
Introduction to Computer Vision
Assignment #2
env:
    Python 3.7.6
    numpy 1.18.1
    opencv-python 4.6.0.66
파일 경로 불러올때 모든 파일이 Assign2/{img_name}.png 와 같은 형식으로 되어있습니다
체점하실때 이 점을 잘 고려해서 해주시면 감사하겠습니다!
'''
def draw_arrowed_line(plane):
    cv2.arrowedLine(plane, (0, 400), (801, 400), (0,0,0), 2, tipLength=0.03)
    cv2.arrowedLine(plane, (400, 801), (400, 0), (0,0,0), 2, tipLength=0.03)
    return plane

def get_M(key):
    '''
    'a' Move to the left by 5 pixels
    ‘d’ Move to the right by 5 pixels
    ‘w’ Move to the upward by 5 pixels
    ‘s’ Move to the downward by 5 pixels

    ‘r’ Rotate counter-clockwise by 5 degrees
    ‘t’ Rotate clockwise by 5 degrees

    ‘f’ Flip across y axis
    ‘g’ Flip across x axis

    ‘x’ Shirnk the size by 5% along to x direction
    ‘c’ Enlarge the size by 5% along to x direction
    ‘y’ Shirnk the size by 5% along to y direction
    ‘u’ Enlarge the size by 5% along to y direction

    ‘h’ Restore to the initial state
    ‘q’ Quit
    '''
    cos_5 = np.cos(np.pi/36)
    sin_5 = np.sin(np.pi/36)
    
    if key in ['a', 'A']:
        M = [
            [1, 0, 0],
            [0, 1, -5],
            [0, 0, 1]
        ]
    elif key in ['d', 'D']:
        M = [
            [1, 0, 0],
            [0, 1, 5],
            [0, 0, 1]
        ]
    elif key in ['w', 'W']:
        M = [
            [1, 0, -5],
            [0, 1, 0],
            [0, 0, 1]
        ]
    elif key in ['s', 'S']:
        M = [
            [1, 0, 5],
            [0, 1, 0],
            [0, 0, 1]
        ]
    
    elif key in ['r', 'R']:
        M = [
            [cos_5, -sin_5, 0],
            [sin_5, cos_5, 0],
            [    0,     0, 1]
        ]
    elif key in ['t', 'T']:
        M = [
            [cos_5, sin_5, 0],
            [-sin_5, cos_5, 0],
            [     0,     0, 1]
        ]
    
    elif key in ['f', 'F']:
        M = [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ]
    elif key in ['g', 'G']:
        M = [
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    
    elif key in ['x', 'X']:
        M = [
            [1, 0, 0],
            [0, 0.95, 0],
            [0, 0, 1]
        ]
    elif key in ['c', 'C']:
        M = [
            [1, 0, 0],
            [0, 1.05, 0],
            [0, 0, 1]
        ]
    elif key in ['y', 'Y']:
        M = [
            [0.95, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    elif key in ['u', 'U']:
        M = [
            [1.05, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    
    elif key in ['h', 'H']:
        M = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    
    else:
        print('잘못된 키 입력으로 인한 종료')
        M = None

    return M

def get_transformed_image(img, M):
    '''
    Implement a function that returns a plane where the transformed image is displayed. The 
    function gets two parameters, an image img and a 3 x 3 affine transformation matrix M. The 
    vertical and horizontal sizes of the plane is fixed to 801 x 801 and the origin (0, 0) is 
    corresponding to the pixel at (400, 400). You also need to draw two arrows to visualize x
    and y axes
    '''

    plane = np.full((801,801), 255)

    half_w = img.shape[0]//2
    half_h = img.shape[1]//2
    for i in range(-half_w, half_w+1):
        for j in range(-half_h, half_h+1):
            coordinates_vector = [i,j,1]
            transformed = np.dot(M, coordinates_vector)

            new_i, new_j = int(transformed[0]) + 400, int(transformed[1]) + 400

            plane[new_i, new_j] = img[i+half_w,j+half_h]
    
    arrowed_plane = draw_arrowed_line(plane)
    return arrowed_plane

if __name__ == '__main__':
    # 체점할때 이미지 파일 경로 수정해야 에러가 안뜹니다!! 감사합니다
    img = cv2.imread('Assign2/smile.png', cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (201,201))
    if img is None:
        print('Image load failed!')
        exit()
    
    M = get_M('h')
    while 1:
        plane = get_transformed_image(img, M)
        cv2.imshow('plane', plane.astype(np.uint8))
        key = cv2.waitKey()
        if key in [ord('q'), ord('Q')]:
            print('Q입력으로 인한 종료')
            break
        elif key in [ord('h'), ord('H')]:
            M = get_M('h')
        else:
            new_M = get_M(chr(key))
            if new_M:
                M = np.dot(new_M,M)
            else:
                break
    cv2.destroyAllWindows()