import cv2
import numpy as np 
from matplotlib import pyplot as plt

testimg = cv2.imread("/Users/jh/Documents/dss/project/ml-repo-2/OCT_small/test/CNV/CNV-3621217-5.jpeg", cv2.IMREAD_COLOR)

#1. CLAHE
def applyclahe(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = cl.apply(src)
    return img 

#2. HE
def applyhe(src):
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(src)
    return img 

#make & save histogram
def makehist(img, filename): 
    plt.hist(img.ravel(), 256, [0,256])
    plt.savefig(filename)
    plt.show()
    return print("complete")

makehist(testimg, "hist.jpeg")
makehist(applyclahe(testimg), "clahehist.jpeg")
makehist(applyhe(testimg), "hehist.jpeg")