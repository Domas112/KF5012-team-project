#%%
from glob import glob
import cv2 as cv
import numpy as np
import os

# path of where the images will be saved
output = "out/"

# for every image in the specified path...
for im in glob('data_path/*.jpg'):
    img = cv.imread(im) # read the image
    
    filename_w_ext = os.path.basename(im) # extract the image name
    filename, file_extension = os.path.splitext(filename_w_ext) # seperate filename from the ".jpg"

    img = cv.resize(img, (256,256), interpolation=cv.INTER_AREA) # resize
    median = cv.medianBlur(img,5) # apply Median filter

    Z = median.reshape((-1,3)) # combine the X and Y axis of the image, leave the colors
    Z = np.float32(Z) # turn to float, as later float is expected as an argument

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5 # how many color centers to look for
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    # now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    kmeans_img = res.reshape((img.shape))

    # create a Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    hsv = cv.cvtColor(kmeans_img, cv.COLOR_RGB2HSV)# convert from RGB to HSV color space

    h, s, v = cv.split(hsv)  # split into 3 different channels
    #apply CLAHE to every channel
    h1 = clahe.apply(h)
    s1 = clahe.apply(s)
    v1 = clahe.apply(v)

    lab = cv.merge((h1,s1,v1))  # merge channels

    enhanced_img = cv.cvtColor(lab, cv.COLOR_LAB2RGB) # convert LAB to BGR
    hsv = cv.cvtColor(enhanced_img, cv.COLOR_RGB2HSV) # convert RGB to HSV for optional use 

    try:
        canny = cv.Canny(enhanced_img, 100, 160) # get edges with the Canny algorithm
        dilated = cv.dilate(canny, (7,7), iterations=3) # dilate the edges, so they may seem thicker
        contours = cv.findContours(dilated,cv.RETR_TREE,cv.CHAIN_APPROX_NONE) # find contours in the dilated version
        cnt = max(contours, key=cv.contourArea) # find the biggest contour

        fill = cv.fillPoly(dilated, pts=[cnt], color=(255,255,255)) # fill the largest found contour
        res = cv.bitwise_and(img, img, mask=fill) # apply the masking onto the original image

        # save the image
        cv.imwrite(os.path.join(output, filename+'.jpg'), res)
        
    except:

        # in case of failure, save the original image
        cv.imwrite(os.path.join(output, filename+'.jpg'), img)

    cv.waitKey(0)
    cv.destroyAllWindows()

# %%
