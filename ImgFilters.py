import cv2
import numpy as np
from skimage.transform import resize


def norm_digit(img):
    m = cv2.moments(img)
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']
    h, w = img.shape[:2]
    aff = np.array([[1, 0, w/2 - cx], [0, 1, h/2 - cy]], dtype=np.float32)
    dst = cv2.warpAffine(img, aff, (0, 0))
    return dst


def mode_change(file, center=False, imgsize = 224, mode = "basic"):
    """
    center : 이미지 중앙에 맞추기 
    -------------------------
    [mode]
    "basic" : 전처리 없음
    "clahe" : CLAHE 적용
    "subtract" : (원본 이미지) - (Gaussian Blur 적용 이미지)
    "DOG" : (Gaussian Blur를 약하게 적용한 이미지) - (Gaussian Blur를 강하게 적용한 이미지)
    * DOG = difference of Gaussians

    """
    if center:
        img = cv2.imread(file, 0)
        img = np.where(img == 255, 0, img)
        img = norm_digit(img)
    else : 
        img = cv2.imread(file, 0)

    if mode == "basic":
        pass
    elif mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
    elif mode == "subtract":
        dst = cv2.GaussianBlur(img, (0, 0), 5)
        img = img - dst
    elif mode == "DOG":
        dst1 = cv2.GaussianBlur(img, (0, 0), 3)
        dst2 = cv2.GaussianBlur(img, (0, 0), 1)
        img = dst2 - dst1

    if center:
        pass
    else:
        img = img/255.0

    img = resize(img, (imgsize, imgsize, 3))
    
    return img
