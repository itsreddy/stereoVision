import scipy.misc
import numpy as np
from matplotlib import pyplot as plt

def corr(img1,img2): 
    return (sum(sum(sum(img1*img2)))/ 
            (sum(sum(sum(img1*img1)))*sum(sum(sum((img2*img2)))))**0.5)

def tempMatch(template,img,blockSize):
    correl = [[0 for i in range(img.shape[1]+blockSize)] for j in range(img.shape[1])]
    for x in range(img.shape[0]-blockSize):
        for y in range(img.shape[1]-blockSize):
            window = img[x:(x+blockSize),y:(y+blockSize)]
            correl[x][y]=corr(window,template)          
    correl = np.array(correl)        
    return np.unravel_index(np.argmax(correl), correl.shape)

img1 = plt.imread('v1.png')
temp = img1[100:110,30:40]
a = tempMatch(temp,img1,10)
print a