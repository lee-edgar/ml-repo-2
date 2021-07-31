from typing import ClassVar
import cv2
import numpy as np 

testimg = cv2.imread("/Users/jh/Documents/dss/project/ml-repo-2/OCT_small/test/CNV/CNV-3621217-5.jpeg", cv2.IMREAD_COLOR)
testimg2 = cv2.imread("/Users/jh/Documents/dss/project/ml-repo-2/pptimg/cat.jpg",cv2.IMREAD_COLOR)


#1. CLAHE 
def applyclahe(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = cl.apply(src)
    return img 

#2. HSV 
def applyhsv(src):
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.inRange(v, 55, 100)
    masking = cv2.bitwise_and(hsv, hsv, mask = v)
    img = cv2.cvtColor(masking, cv2.COLOR_HSV2BGR)
    return img

#3. DoG 
def applydog(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst1 = cv2.GaussianBlur(src, (0, 0), 3)
    dst2 = cv2.GaussianBlur(src, (0, 0), 1)
    img = dst2 - dst1
    return img

#4. Subtraction 
def applysubtract(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.GaussianBlur(src, (0, 0), 5)
    img = src - dst
    return img

#5. Contour Mask
def applycontour(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(src, 80, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(src.shape, np.uint8)
    cv2.drawContours(mask, contours, -1, (255), 1)
    return mask

#check and save
def showandsave(img, savedir):
    cv2.imshow("result", img)
    cv2.imwrite(savedir, img)
    cv2.waitKey(0)
    return print("complete")

#Gaussians
def gaussians(src) : 
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(src, (0, 0), 1)
    g2 = cv2.GaussianBlur(src, (0, 0), 3)
    g3 = cv2.GaussianBlur(src, (0, 0), 5)

    for num, i in enumerate([g1, g2, g3]):
        num = num * 2 + 1
        savedir = "pptimg/gaussian-" + str(num) + ".jpeg"
        cv2.imwrite(savedir, i)
    return print("complete")

#showandsave(applyclahe(testimg), "pptimg/clahe.jpeg")
#showandsave(applyhsv(testimg), "pptimg/hsv.jpeg")
#showandsave(applydog(testimg), "pptimg/dog.jpeg")
#showandsave(applysubtract(testimg), "pptimg/subtract.jpeg")
#showandsave(applycontour(testimg), "pptimg/contour.jpeg")
#showandsave(applyclahe(testimg2), "pptimg/clahecat.jpeg")

