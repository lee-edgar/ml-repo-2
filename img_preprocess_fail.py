import cv2
import numpy as np 

testimg = cv2.imread("/Users/jh/Documents/dss/project/ml-repo-2/OCT_small/test/CNV/CNV-3621217-5.jpeg", cv2.IMREAD_COLOR)
testimg2 = cv2.imread("/Users/jh/Documents/dss/project/ml-repo-2/pptimg/cat.jpg",cv2.IMREAD_COLOR)

#1. Canny Edge
def applycanny(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(src, 10 , 30, 1) 
    return img 

#2. HE
def applyhe(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(src)
    return img 
    

#3. NlMeans
def applynlmeans(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.fastNlMeansDenoising(src, None, 16, 7, 21)
    return img 

#4. bilateral
def applybi(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(src,9,75,75)
    return img 

#5. Gaussian for Comparison
def applygaussian(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(src, (0, 0), 3)
    return img    

#check and save
def showandsave(img, savedir):
    cv2.imshow("result", img)
    cv2.imwrite(savedir, img)
    cv2.waitKey(0)
    return print("complete")

# showandsave(applycanny(testimg), "pptimg/canny.jpeg")
# showandsave(applyhe(testimg), "pptimg/he.jpeg")
# showandsave(applynlmeans(testimg), "pptimg/nlmeans.jpeg")
# showandsave(applybi(testimg), "pptimg/bilateral.jpeg")
# showandsave(applygaussian(testimg), "pptimg/gaussian.jpeg")

showandsave(applyhe(testimg2), "pptimg/hecat.jpeg")



