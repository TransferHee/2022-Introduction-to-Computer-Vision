import cv2
import numpy as np
import time
import os

from assign_1_utils import get_image, cv_imshow
'''
2018314848 한승희
Introduction to Computer Vision
Assignment #1
env:
    Python 3.7.6
    numpy 1.18.1
    opencv-python 4.6.0.66
'''
def padding(img, kernel_size, padding_type = '2d'):
    
    # case vertical concat
    if padding_type == 'V':
        for _ in range(kernel_size//2):
            img = np.concatenate((img[0,:].reshape(1,-1), img, img[-1,:].reshape(1,-1)))

    # case horizontal concat
    elif padding_type == 'H':
        for _ in range(kernel_size//2):
            img = np.concatenate((img[:,0].reshape(-1,1), img, img[:,-1].reshape(-1,1)), axis=1)

    # case 2d img
    elif padding_type == '2d':
        for _ in range(kernel_size//2):
            img_2 = np.concatenate((img[0,:].reshape(1,-1), img, img[-1,:].reshape(1,-1)))
            img = np.concatenate((img_2[:,0].reshape(-1,1), img_2, img_2[:,-1].reshape(-1,1)), axis=1)
            
    return img

def cross_correlation_1d(img, kernel):
    '''
     Your function cross_correlation_1d should distinguish between vertical and horizontal
     kernels based on the shape of the given kernel.
    
    You can assume that the all kernels are odd sized along both dimensions.
    
    Your functions should preserve the size of the input image. In order words, the sizes of img
    and filtered_img should be identical. To do this, you need to handle boundary cases on the 
    edges of the image. Although you can take various approaches, you are asked to pad the image 
    such that pixels lying outside the image have the same intensity value as the nearest pixel inside 
    the image.
    (have the same intensity value as the nearest pixel inside the image padding)
    
    You cannot use any built-in function that performs cross-correlation, colvolution, filtering or 
    image padding.
    '''

    output = np.zeros(img.shape)
    kernel_size = kernel.shape[0]
    if kernel.ndim == 1:
        # case horizontal kernel
        # print('ho')
        padding_img = padding(img, kernel_size, padding_type='H')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cropped_row = padding_img[i,j:j+kernel_size]
                results = cropped_row * kernel
                output[i,j] = results.sum()

    else:
        # case vertical kernel
        # print('ve')
        kernel = np.squeeze(kernel)
        padding_img = padding(img, kernel_size, padding_type='V')

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cropped_col = padding_img[i:i+kernel_size, j]
                results = cropped_col * kernel
                output[i,j] = results.sum()

    return output

def cross_correlation_2d(img, kernel):
    output = np.zeros(img.shape)
    kernel_size = kernel.shape[0]
    padding_img = padding(img, kernel_size, padding_type = '2d')
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cropped_img = padding_img[i:i+kernel_size, j:j+kernel_size]
            results = cropped_img * kernel
            output[i,j] = results.sum()
            
    return output

def gaussian_func(i, j, sigma):
    x = -(i**2 + j**2) / (2*(sigma**2))
    return np.exp(x) / (2*np.pi*(sigma**2))

def get_gaussian_filter_1d(size, sigma):
    '''
    You can assume that size is an odd number.
    
    Print the results of get_gaussian_filter_1d(5,1) and get_gaussian_filter_2d(5,1) to 
    the console.
    
    Perform at least 9 different Gaussian filtering to an image (e.g., combinations of 3 different 
    kernel sizes and sigma values). Show the filtered images in a single window and write them as 
    a single image file ‘./result/part_1_gaussian_filtered_{INPUT_IMAGE_FILE_NAME}’. You 
    are also asked to display a text caption describing the filter parameters on each filtered image.
    
    Perform the Gaussian filtering by applying vertical and horizontal 1D kernels sequantially, and 
    compare the result with a filtering with a 2D kernel. Specifically, visualize a pixel-wise 
    difference map and report the sum of (absolute) intensity differences to the console. You are 
    also required to report the computational times of 1D and 2D filterings to the console. Note 
    that, you can report one of above 9 different cases (i.e. 17x17 s=6).
    
    Your script should produce results of (d) and (e) for ‘lenna.png’ and ‘shapes.png’.
    (i.e. ‘part_1_gaussian_filtered_lenna.png’ and ‘part_1_gaussian_filtered_shapes.png’)
    
    When performing filtering, you have to use the function implemented in 1-1.
    
    You cannot use any built-in function that produces Gaussian filters.
    '''
    
    kernel = np.ones(size)
    kernel = np.array([gaussian_func(i - size//2, 0, sigma) for i in range(size)])
    kernel /= kernel.sum()
    return kernel

def get_gaussian_filter_2d(size, sigma):
    kernel = np.ones([size, size])
    kernel = np.array([[gaussian_func(i - size//2, j - size//2, sigma) for j in range(size)] for i in range(size)])
    kernel /= kernel.sum()
    return kernel

def show_different_filtered_images(img_name, size_lst=[5,11,17], sig_lst=[1,6,11]):
    '''
    Perform at least 9 different Gaussian filtering to an image (e.g., combinations of 3 different 
    kernel sizes and sigma values). Show the filtered images in a single window and write them as 
    a single image file ‘./result/part_1_gaussian_filtered_INPUT_IMAGE_FILE_NAME’. You 
    are also asked to display a text caption describing the filter parameters on each filtered image.
    '''
    
    img = get_image(img_name)
    output_img = None
    first_img = True
    for size in size_lst:
        tmp = None
        first_tmp = True
        for sig in sig_lst:
            kernel = get_gaussian_filter_2d(size, sig)
            filtered_img = cross_correlation_2d(img, kernel)
            img_with_text = cv2.putText(
                img=filtered_img,
                text=f'{size}x{size} s={sig}', 
                org=(10, 40), 
                fontFace= cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0,0,0),
                thickness=2)

            if first_tmp:
                tmp = img_with_text
                first_tmp = False
            else:
                tmp = np.concatenate((tmp, img_with_text), axis=1)


        if first_img:
            output_img = tmp
            first_img = False
        else:
            output_img = np.concatenate((output_img, tmp))

    # 왜인지 모르겠는데 np.uint8로 안하면 이미지가 흰색으로 띈다
    output_img = output_img.astype(np.uint8)
    # 그냥 출력하면 9개(혹은 그 이상)사진의 크기가 너무 커져서 반절로 줄인 후 출력
    output_img = cv2.resize(output_img, dsize=(0,0), fx=0.5, fy=0.5)
    cv2.imwrite(f"./result/part_1_gaussian_filtered_{img_name}.png",output_img)
    cv_imshow(f'part_1_gaussian_filtered_{img_name}', output_img)


if __name__ == "__main__":

    # result폴더가 없다면 디렉토리를 생성(이미 있으면 무시)
    os.makedirs('./result', exist_ok=True)

    img_name = input('Which image do you want? (without .png): ')
    target_img = get_image(img_name)

    # 1-2 c)
    print('\nGaussian_filter_1D (size=5, sigma=1)')
    print(get_gaussian_filter_1d(5,1))
    print('\nGaussian_filter_2D(size=5,sigma=1)')
    print(get_gaussian_filter_2d(5,1))

    # 1-2 d)
    show_different_filtered_images(img_name)

    # 1-2 e)
    size, sigma = 17, 6

    kernel_1D = get_gaussian_filter_1d(size, sigma)
    start_time = time.time()
    vertical_filtered_img = cross_correlation_1d(target_img, kernel_1D)
    filtered_img_1D = cross_correlation_1d(vertical_filtered_img, kernel_1D.reshape(-1,1))
    print('\n1d cross-correlation time:',time.time() - start_time)

    kernel_2D = get_gaussian_filter_2d(size, sigma)
    start_time = time.time()
    filtered_img_2D = cross_correlation_2d(target_img, kernel_2D)
    print('2d cross-correlation time:',time.time() - start_time)

    difference_map = filtered_img_1D - filtered_img_2D
    print('\nDifference map between 1D filtering and 2D filtering:\n',difference_map)
    print('Absolute sum of intensity differences:',np.abs(difference_map).sum())
    cv_imshow('Difference map', difference_map)
    # print('abs', np.abs(difference_map))
    # print('sum', np.abs(difference_map).sum())
