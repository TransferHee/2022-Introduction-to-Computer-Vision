import cv2

def get_image(name):
    img_path = name + '.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img

def cv_imshow(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey()

    cv2.destroyAllWindows()