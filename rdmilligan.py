import cv2
import numpy as np

def show(imx, winName="test"):
    cv2.imshow(winName, imx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
# disparity settings
window_size = 5
min_disp = 32
num_disp = 112-min_disp
stereo = cv2.StereoSGBM(
    minDisparity = min_disp,
    numDisparities = num_disp,
    SADWindowSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 1,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    fullDP = False
)
 
 # load stereo image

image_left = cv2.imread('images/im_left.jpg')
image_right = cv2.imread('images/im_right.jpg')

# compute disparity
# disparity = stereo.compute(image_left, image_right).astype(np.float32) / 16.0
# disparity = (disparity-min_disp)/num_disp

disp = np.zeros_like(image_left)

stereo.compute(image_left, image_right, disp)

show(disp)