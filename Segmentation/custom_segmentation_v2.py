import cv2 as cv
import numpy as np
from glob import glob
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

    # specify the color bounds
    lower_green = np.array([50,100,100])
    upper_green = np.array([100,255,255])
    # obtain the mask based on the bounds
    mask_g = cv.inRange(hsv, lower_green, upper_green)

    # inverted mask
    _, inv_mask = cv.threshold(mask_g,127,255,cv.THRESH_BINARY_INV)
    # apply the mask onto the image
    res = cv.bitwise_and(img,img, mask= mask_g)

    mask = np.zeros(img.shape[:2],np.uint8)
    bck_model = np.zeros((1,65),np.float64)
    frg_model = np.zeros((1,65),np.float64)


    if (np.sum(inv_mask) < (256*256*255)):
        print("not using rectangle")
        newmask = inv_mask
        mask[newmask == 0] = 0
        mask[newmask == 255] = 1
        cv.grabCut(img,mask,None,bck_model,frg_model,5,cv.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        segment_img = img*mask2[:,:,np.newaxis]


    else:
        print("using rectangle")
        s = (img.shape[0] / 10, img.shape[1] / 10)
        rect = np.array((s[0], s[1], img.shape[0] - (3/10) * s[0], img.shape[1] - s[1])).astype("uint8")
        cv.grabCut(lab,mask,rect,bck_model,frg_model,10,cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        segment_img= img*mask2[:,:,np.newaxis]

    cv.imwrite(os.path.join(output, filename+'.jpg'), segment_img)
    cv.destroyAllWindows()