import cv2
import numpy as np
import time
from assign_1_utils import get_image, cv_imshow
from A1_image_filtering import cross_correlation_2d, get_gaussian_filter_2d
from A1_edge_detection import get_sobel_filter

def update_and_normalize(matrix):
    matrix[matrix < 0] = 0
    return (matrix - matrix.min()) / (matrix.max() - matrix.min())

def compute_corner_response(img):
    '''
    a)First apply the Sobel filters to compute derivatives along x and y directions
    
    b) For each pixel, compute the second moment matrix M. You can utilize an uniform window 
    function (i.e. w(x, y) = 1 if (x, y) is lying in the window, otherwise w(x, y) = 0).
    Use 5 x 5 window to compute the matrix M.

    c) You should use the following response function with 𝜅 = 0.04:
    R = 𝜆1𝜆2 - 𝜅(𝜆1+𝜆2)^2

    d) Once the response values for all the pixels are computed, update all the negative responses to 0 
    and then normalize them to a range of [0,1].

    e) Report the computational time of compute_corner_response to the console.
    Visualize the computed corner response intensities and store them as an image file
    ‘./result/part_3_corner_ raw_ INPUT_IMAGE_FILE_NAME’.

    f) Your script should produce (d) for ‘shapes.png’ and ‘lenna.png'
    '''
    output = np.zeros(img.shape)
    window_size = 5
    offset = window_size // 2
    k = 0.04

    sobel_x, sobel_y = get_sobel_filter()

    I_x, I_y = cross_correlation_2d(img, sobel_x), cross_correlation_2d(img, sobel_y)
    IxIx = I_x ** 2
    IxIy = I_x * I_y
    IyIy = I_y ** 2

    for i in range(offset, img.shape[0]-offset):
        for j in range(offset, img.shape[1]-offset):
            A = np.sum(IxIx[i-offset:i+offset+1, j-offset:j+offset+1])
            B = np.sum(IxIy[i-offset:i+offset+1, j-offset:j+offset+1])
            C = np.sum(IyIy[i-offset:i+offset+1, j-offset:j+offset+1])

            # M = np.array([
            #     [A, B],
            #     [B, C]
            # ])

            det_M = A*C - B**2
            trace_M = A+C
            output[i,j] = det_M - k*(trace_M**2)

    output = update_and_normalize(output)
    return output

def show_corner_as_green(img_name, target_img, corner_response):
    '''
    a) Change the color of pixels having corner response greater than a threshold of 0.1 to green.

    b) Display the result of (a) and store it as an image file 
    ‘./result/part_3_corner_bin_ INPUT_IMAGE_FILE_NAME’
    '''

    color_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)
    color_img[corner_response > 0.1] = [0, 255, 0]

    cv_imshow(f'part_3_corner_bin_{img_name}', color_img)
    cv2.imwrite(f'./result/part_3_corner_bin_{img_name}.png', color_img)

def non_maximum_suppression_win(R, winSize=11):
    '''
    This function suppresses (i.e. set to 0) the corner response at a position (x, y) if it is not a 
    maximum value within a squared window sized winSize and centered at (x, y). Althogh the 
    response is a local maxima, it is suppressed if it not greater than the threshold 0.1.
    Set the parameter winSize = 11.
    '''

    offset = winSize // 2
    output = np.zeros(R.shape)

    for i in range(offset, R.shape[0]-offset+1):
        for j in range(offset, R.shape[1]-offset+1):
            center = R[i,j]
            cropped = R[i-offset:i+offset+1, j-offset:j+offset+1]

            output[i,j] = np.where(center <= 0.1 or center < cropped.max(), 0, R[i,j])

    return output

def draw_green_circles(img_name, target_img, suppressed_R):

    color_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)
    
    for i in range(suppressed_R.shape[0]):
        for j in range(suppressed_R.shape[1]):
            if suppressed_R[i,j] != 0:
                cv2.circle(color_img, (j,i), 5, (0,255,0), 2)
    
    cv_imshow(f'part_3_corner_sup_{img_name}', color_img)
    cv2.imwrite(f'./result/part_3_corner_sup_{img_name}.png', color_img)

if __name__ == '__main__':

    # 3-1)
    img_name = input('Which image do you want? (without .png): ')
    # img_name = 'shapes'
    # img_name = 'lenna'
    target_img = get_image(img_name)

    gaussian_filter = get_gaussian_filter_2d(7, 1.5)
    filtered_img = cross_correlation_2d(target_img, gaussian_filter)

    # 3-2)
    start = time.time()
    corner_response = compute_corner_response(filtered_img)

    print('Compution time of raw corner detection:',time.time() - start)

    cv_imshow(f'part_3_corner_raw_{img_name}', corner_response)
    cv2.imwrite(f'./result/part_3_corner_raw_{img_name}.png', corner_response)

    # 3-3)
    show_corner_as_green(img_name, target_img, corner_response)

    start = time.time()
    suppressed_R = non_maximum_suppression_win(corner_response)

    print('Compution time of corner response suppression:',time.time() - start)

    draw_green_circles(img_name, target_img, suppressed_R)