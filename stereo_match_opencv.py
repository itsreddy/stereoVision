'''
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def show(imx, winName="test"):
    cv2.imshow(winName, imx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('loading images...')
    img = cv2.imread('images/trial2.jpg')

    print(img.shape)
    imgL = img[:, :960]
    imgR = img[:, 960:] 
    # imgL = cv2.imread('images/w1.jpg')
    # imgR = cv2.imread('images/w2.jpg')


    imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    # disparity range is tuned for 'aloe' image pair
    # window_size = 3
    # min_disp = 32
    # num_disp = 112-min_disp
    # stereo = cv.StereoSGBM_create(

    #     minDisparity = min_disp,
    #     numDisparities = num_disp,
    #     blockSize = 15,
    #     P1 = 8*3*window_size**2,
    #     P2 = 32*3*window_size**2,
    #     disp12MaxDiff = 1,
    #     uniquenessRatio = 10,
    #     speckleWindowSize = 100,
    #     speckleRange = 32

    # )

    # print('computing disparity...')
    # disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0



    # cv.imshow('left', imgL)
    # cv.imshow('disparity', (disp-min_disp)/num_disp)
    # cv.waitKey()
    # cv.destroyAllWindows()

    blockSize = 40

    stereo = cv2.StereoSGBM_create(minDisparity=1,
        numDisparities=16,
        blockSize=15,
        uniquenessRatio = 10,
        speckleWindowSize = 10,
        speckleRange = 32,
        disp12MaxDiff = 1,
        P1 = 8*3*blockSize**2,
        P2 = 32*3*blockSize**2
    )

    disparity = stereo.compute(imgL_gray, imgR_gray)
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    # # Normalize the image for representation
    # min = disparity.min()
    # max = disparity.max()
    # disparity = np.uint8(255 * (disparity - min) / (max - min))

    # Display the result
    # show(np.hstack((imgL, imgR)))
    show(disparity)

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')