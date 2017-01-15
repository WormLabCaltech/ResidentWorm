# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 16:28:13 2016

@author: Cynthia
"""
def calibration(mp4filename, rad, Q1x, Q1y, Q2x, Q2y, Q3x, Q3y, Q4x, Q4y):
    """
    Function accepts an mp4 video file and fits quadrant-shaped masks to the 
    first video frame. Control and Experimental ROI masks are each made up of 
    two diagonally opposing quadrants. 
    
    Parameters:
        mp4filename: name of calibration video file 
        rad: 
        Q1x: x-pixel value for upper left quadrant center 
        Q1y: y-pixel value for upper left quadrant center 
        Q2x: x-pixel value for upper right quadrant center 
        Q2y: y-pixel value for upper right quadrant center 
        Q3x: x-pixel value for lower left quadrant center 
        Q3y: y-pixel value for lower left quadrant center 
        Q4x: x-pixel value for lower right quadrant center 
        Q4y: y-pixel value for lower right quadrant center 
    
    Output:
        New Window (press any key to move through the following windows):
            Video frame for calibration
            Video frame with control ROI mask applied
            Video frame with experimental ROI mask applied
            
        In Console:
            Total number of pixels in control ROI
            Total number of pixels in experimental ROI
    
    """
    import cv2
    import numpy as np
    
    # read video file
    vidcap = cv2.VideoCapture(mp4filename)
    count = 0
    
    # read first frame of calibration video
    while count < 1:
      success, img = vidcap.read()
      count += 1
    
    # convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # build white mask against black background for control quadrants
    mask1 = np.zeros(img.shape, dtype = np.uint8)     
    cv2.ellipse(mask1, (Q1x, Q1y), (rad, rad), 0, 180, 270, 255, -1) 
    mask4 = np.zeros(img.shape, dtype = np.uint8)     
    cv2.ellipse(mask4, (Q4x, Q4y), (rad, rad), 0, 0, 90, 255, -1)
    mask_ctrl = mask1 + mask4
    
    # build white mask against black background for experimental quadrants
    mask2 = np.zeros(img.shape, dtype = np.uint8)     
    cv2.ellipse(mask2, (Q2x, Q2y), (rad, rad), 0, 270, 360, 255, -1) 
    mask3 = np.zeros(img.shape, dtype = np.uint8)      
    cv2.ellipse(mask3, (Q3x, Q3y), (rad, rad), 0, 90, 180, 255, -1) 
    mask_exp = mask2 + mask3
    
    # use bitwise & operator to apply transparent mask to calibration frame
    roi_ctrl = img & mask_ctrl   
    roi_exp = img & mask_exp
    
    # show calibration video frame 
    cv2.imshow('img', img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # show video frame with control mask applied 
    cv2.imshow('roi_ctrl', roi_ctrl) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # show video frame with experimental mask applied 
    cv2.imshow('roi_exp', roi_exp) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # assign variables to pixel width and length of video frame
    y_pix, x_pix = img.shape 
    
    # count number of pixels in control mask
    count_mask_ctrl = 0                                       
    for i in range(y_pix):          
        for j in range(x_pix):
            if mask_ctrl[i][j] == 255:
                count_mask_ctrl += 1
            else:
                count_mask_ctrl += 0
    
    # count number of pixels in experimental mask
    count_mask_exp = 0                                       
    for i in range(y_pix):          
        for j in range(x_pix):
            if mask_exp[i][j] == 255:
                count_mask_exp += 1
            else:
                count_mask_exp += 0
    
    # print total number of pixels in control mask and in experimental mask
    print ('Total number of pixels in Control ROI = ', count_mask_ctrl)
    print ('Total number of pixels in Experimental ROI = ', count_mask_exp)
        
    return;