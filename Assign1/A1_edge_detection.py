import cv2
import numpy as np
import time
from A1_image_filtering import cross_correlation_2d, get_gaussian_filter_2d, padding
from assign_1_utils import get_image, cv_imshow

def get_sobel_filter():
    filter_x = np.array([
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]
    ])
    filter_y = np.array([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1]
    ])
    return filter_x, filter_y

def get_magnitude(x, y):
    return np.sqrt(x**2 + y**2)

def compute_image_gradient(img):
    '''
    a) First apply the Sobel filters to compute derivatives along x and y directions

    b) For each pixel, compute magnitude and direction of gradient.

    c) You need to use functions implemented in Part #1.

    d) Report the computational time taken by compute_image_gradient to the console.
    Show the computed manitude map and store it to an image file 
    ‘./result/part_2_edge_raw_ INPUT_IMAGE_FILE_NAME’.

    e) Your script should produce (d) for ‘shapes.png’ and ‘lenna.png
    '''

    sobel_filter_x, sobel_filter_y = get_sobel_filter()

    x_derivatives = cross_correlation_2d(img, sobel_filter_x)
    y_derivatives = cross_correlation_2d(img, sobel_filter_y)

    dir = np.arctan2(y_derivatives, x_derivatives)
    mag = get_magnitude(x_derivatives, y_derivatives)

    return mag, dir

def dir_quantized(ang):
    if (ang <= 22.5) or (ang > 337.5) or (157.5 < ang <= 202.5):
        case = 1
    elif (22.5 < ang <= 67.5) or (202.5 < ang <= 247.5):
        case = 2
    elif (67.5 < ang <= 112.5) or (247.5 < ang <= 292.5):
        case = 3
    elif (112.5 < ang <= 157.5) or (292.5 < ang <= 337.5):
        case = 4
    else:
        print('내가 놓친게 있나?', ang)
    return case
    
def non_maximum_suppression_dir(mag, dir):
    '''
    a) You are asked to implement an approximated version of NMS by quantizing the gradient 
    directions into 8 bins. If a direction is represented by an angle in degrees, we can map the 
    direction to the closet representative angle among [0°, 45°, … ,315°].

    b) Once the direction is quantized, compare the gradient magnitude against two magnitudes along 
    the quantized direction. In this assignment, you do not have to interpolate the magnitude for the 
    simplicity. If the gradient magnitude at the center position is not greater than the ones along the 
    gradient direction, it is suppressed (the magnitude is set to 0).

    c) For instance, if the gradient direction is 145° then it is quantized to 135°. In this case, the 
    magnitude at the center position in the window should be compared to the north-west and south-east positions as illustrated in the figure below.
    
    d) Report the computational time consumed by non_maximum_suppression_dir to the console.
    Show the supressed manitude map and store it to an image file 
    ‘./result/part_2_edge_sup_ INPUT_IMAGE_FILE_NAME’.
    
    e) Your script should produce (d) for ‘shapes.png’ and ‘lenna.png’

    '''
    
    angles = np.degrees(dir) + 180
    output = np.zeros(mag.shape)
    padding_mag = padding(mag, 3, padding_type='2d')

    for i in range(1, padding_mag.shape[0]-1):
        for j in range(1, padding_mag.shape[1]-1):
            case = dir_quantized(angles[i-1,j-1])
            if case == 1:
                # case 1 ㅡ compare
                output[i-1,j-1] = np.where((padding_mag[i,j] >= max(padding_mag[i,j-1], padding_mag[i,j+1])), padding_mag[i,j], 0)
                
            elif case == 2:
                # case 2 / compare
                output[i-1,j-1] = np.where((padding_mag[i,j] >= max(padding_mag[i-1,j-1], padding_mag[i+1,j+1])), padding_mag[i,j], 0)
                
            elif case == 3:
                # case 3 | compare
                output[i-1,j-1] = np.where((padding_mag[i,j] >= max(padding_mag[i-1,j], padding_mag[i+1,j])), padding_mag[i,j], 0)

            else:
                # case 4 \ compare
                output[i-1,j-1] = np.where((padding_mag[i,j] >= max(padding_mag[i-1,j+1], padding_mag[i+1,j-1])), padding_mag[i,j], 0)
    
    return output

if __name__ == '__main__':

    # 2-1
    img_name = input('Which image do you want? (without .png): ')
    # img_name = 'shapes'
    # img_name = 'lenna'
    target_img = get_image(img_name)

    gaussian_filter = get_gaussian_filter_2d(7, 1.5)
    filtered_img = cross_correlation_2d(target_img, gaussian_filter)

    # 2-2
    start = time.time()
    mag, dir = compute_image_gradient(filtered_img)

    print('Compution time of raw edge detection:',time.time() - start)

    clip_mag = np.clip(mag, 0.0, 255.0)
    clip_mag = clip_mag.astype(np.uint8)
    cv_imshow(f'part_2_edge_raw_{img_name}', clip_mag)
    cv2.imwrite(f'./result/part_2_edge_raw_{img_name}.png', clip_mag)

    # 2-3
    start = time.time()
    suppressed_mag = non_maximum_suppression_dir(mag, dir)

    print('Compution time of non_maximum_suppression:',time.time() - start)

    clip_sup_mag = np.clip(suppressed_mag, 0.0, 255.0)
    clip_sup_mag = clip_sup_mag.astype(np.uint8)
    cv_imshow(f'part_2_edge_sup_{img_name}', clip_sup_mag)
    cv2.imwrite(f'./result/part_2_edge_sup_{img_name}.png', clip_sup_mag)
    