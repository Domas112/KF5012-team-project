import cv2 as cv
import numpy as np
from glob import glob
import os

output = "out/"

for im in glob('data_path/*.jpg'):
    
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

    hsv = cv.cvtColor(enhanced_img, cv.COLOR_BGR2HSV)

    lower_green = np.array([50,100,100])
    upper_green = np.array([100,255,255])
    mask_g = cv.inRange(hsv, lower_green, upper_green)

    _, inv_mask = cv.threshold(mask_g,127,255,cv.THRESH_BINARY_INV)
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


    # cv.imshow("original", img)
    # cv.imshow("hsv", hsv)
    # cv.imshow("kmeans", kmeansimg)
    # cv.imshow("segmented", segmentedimg)
    # cv.imshow("grapcut", Segmented_mask)
    # cv.waitKey(0)

    
    # segment_img_gray = cv.cvtColor(segment_img, cv.COLOR_RGB2GRAY)

    # _,thresh = cv.threshold(segment_img_gray,1,255,cv.THRESH_BINARY)

    # contours,hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]
    # x,y,w,h = cv.boundingRect(cnt)
    
    # crop = segment_img_gray[y:y+h,x:x+w]
    # cv.imwrite('sofwinres.png',crop)


    cv.imwrite(os.path.join(output, filename+'.jpg'), segment_img)
    cv.destroyAllWindows()