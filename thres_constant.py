# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 21:57:28 2017

@author: Cynthia
"""

def threshold_constant( filename, frames, constant):
    """
    
    Function accepts an mp4 video file and outputs a specified number of 
    thresholded video frames sequentially. Threshold value is adjusted by
    changing the 'constant' parameter.
    
    Parameters:
        mp4filename: name of video file 
        frames: number of video frames to be viewed (enter a number from 1 to 
                number of frames in video)
        constant: constant value subtracted from threshold value 
                  (5-7 recommended)
       
    Output:
        New window(s) showing thresholded video frame(s). Hit any key to view 
        specified number of frames sequentially. 
            
    """
    
    import cv2
    
    vidcap = cv2.VideoCapture(filename)          
    success,image = vidcap.read() 
    
    count = 0; 
                                                     
    while count < frames:    
        success,image = vidcap.read() 
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV,11,constant)
        
        cv2.imshow('img', img) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        count+=1
        
    return;
    
    