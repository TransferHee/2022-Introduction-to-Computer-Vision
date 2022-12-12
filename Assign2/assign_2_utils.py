import cv2

def get_image(name):
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    return img

def cv_imshow(img_name, img):
    cv2.imshow(img_name, img)
    cv2.waitKey()

    cv2.destroyAllWindows()