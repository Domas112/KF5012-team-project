#%%
from glob import glob
import cv2 as cv
import numpy as np
import os


output = "out/"


for im in glob('data_path/*.jpg')[:1]:
    img = cv.imread(im)
    filename_w_ext = os.path.basename(im)
    filename, file_extension = os.path.splitext(filename_w_ext)
    print(filename)

    img = cv.resize(img, (256,256), interpolation=cv.INTER_AREA)
    median = cv.medianBlur(img,5) # Apply Median filter

    Z = median.reshape((-1,3))
    Z = np.float32(Z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    kmeans_img = res.reshape((img.shape))

    clahe = cv.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    hsv = cv.cvtColor(kmeans_img, cv.COLOR_BGR2HSV)# convert from BGR to HSV color space

    h, s, v = cv.split(hsv)  # split on 3 different channels
    #apply CLAHE to the L-channel
    h1 = clahe.apply(h)
    s1 = clahe.apply(s)
    v1 = clahe.apply(v)

    lab = cv.merge((h1,s1,v1))  # merge channels

    enhanced_img = cv.cvtColor(lab, cv.COLOR_LAB2BGR)  # convert from LAB to BGR
    # hsv = cv.cvtColor(enhanced_img, cv.COLOR_BGR2HSV)
    # hsvrgb = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    gray = cv.cvtColor(kmeans_img, cv.COLOR_RGB2GRAY)

    _, threshold = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # _, threshold = cv.threshold(hsv,0,255,cv.THRESH_BINARY_INV)


    contours, hierarchy = cv.findContours(threshold,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv.contourArea)

    # Create a new mask for the result image
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Perform closing to remove hair and blur the image
    # kernel = np.ones((15,15),np.uint8) 

    # closing = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel, iterations = 2)
    # blur = cv.blur(closing,(15,15))

    # Draw the contour on the new mask and perform the bitwise operation
    cv.drawContours(mask, [cnt],-1, 255, -1)
    res = cv.bitwise_and(img, img, mask=mask)

    # cv.imwrite(os.path.join(output, filename+'.jpg'), res)
    # cv.destroyAllWindows()

    cv.imshow("kmeans", kmeans_img)
    cv.imshow("gray", gray)
    # cv.imshow("hsv", hsv)
    # cv.imshow("hsvrgb", hsvrgb)
    # cv.imshow("ench", enhanced_img)
    cv.imshow("thresh", threshold)
    cv.imshow("final", res)
    cv.waitKey(0)
    cv.destroyAllWindows()



# %%
