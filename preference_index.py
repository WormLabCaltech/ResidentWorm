# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:19:20 2016

@author: Cynthia
"""

def preference_index(mp4filename, csvfilename, pngplot, Q1x, Q1y, Q2x, Q2y, 
                     Q3x, Q3y, Q4x, Q4y, rad, ROI_size, constant):
    """
    
    Function accepts an mp4 video file and counts the number of worm pixels in 
    each of the control and experimental regions of interest (ROIs) for each 
    video frame. These pixel values are then used to calculate a preference 
    index for the experimental cue using the following formula: 
    Preference Index (PI) = (exp-ctrl)/(exp+ctrl). Outputs a csv file and a png 
    figure.
    
    Parameters:
        mp4filename: name of mp4 video file (include quotation marks)
        csvfilename: name of csv output file (include quotation marks)
        pngplot: name of worm occupancy plot (include quotation marks)
        Q1x: x-pixel value for upper left quadrant center 
        Q1y: y-pixel value for upper left quadrant center 
        Q2x: x-pixel value for upper right quadrant center 
        Q2y: y-pixel value for upper right quadrant center 
        Q3x: x-pixel value for lower left quadrant center 
        Q3y: y-pixel value for lower left quadrant center 
        Q4x: x-pixel value for lower right quadrant center 
        Q4y: y-pixel value for lower right quadrant center 
        rad: radius of circle quadrant
        ROI_size: total number of pixels in each ROI 
                (equivalent for control and experimental ROIs)
        constant: thresholding constant (3-4 recommended)
         
    Output:
        csv file: records worm pixel values in each ROI and preference index 
                  per video frame
        png figure: visualizes worm occupancy per ROI throughout video with
                    Preference Index as title
    
    """
    import cv2
    import numpy as np    
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_style('ticks')
    
    # play mp4 video                        
    vidcap = cv2.VideoCapture(mp4filename)                                      
    
    # count number of frames in video
    frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)                        
    
    # read first video frame and convert to grayscale
    shape_count = 0
    
    while shape_count < 1:
      success, calib_img = vidcap.read()
      shape_count += 1
    
    calib_img = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)
    
    # generate control mask using values from calibration script
    mask1 = np.zeros(calib_img.shape, dtype = np.uint8)                           
    cv2.ellipse(mask1,(Q1x, Q1y),(rad, rad), 0, 180, 270, 255, -1) 
    mask4 = np.zeros(calib_img.shape, dtype = np.uint8)     
    cv2.ellipse(mask4,(Q4x, Q4y),(rad, rad), 0, 0, 90, 255, -1)
    mask_ctrl = mask1 + mask4
    
    # generate experimental mask using values from calibration script
    mask2 = np.zeros(calib_img.shape, dtype = np.uint8)                             
    cv2.ellipse(mask2, (Q2x, Q2y), (rad, rad), 0, 270, 360, 255, -1) 
    mask3 = np.zeros(calib_img.shape, dtype = np.uint8)      
    cv2.ellipse(mask3, (Q3x, Q3y), (rad, rad), 0, 90, 180, 255, -1) 
    mask_exp = mask2 + mask3
    
    # assign variables to total number of pixels in each ROI    
    count_mask_ctrl = ROI_size                                                  
    count_mask_exp = ROI_size               
    
    # create empty lists to input worm pixel values in each ROI per frame
    pix_count_ctrl = []                                                         
    pix_count_exp = []
    
    # frame counter
    count = 0
    
    # counts number of worm pixels in each ROI per frame                                                                            
    while count < frames:    
        success,image = vidcap.read()
        
        # convert RGB images to grayscale 
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        
        # apply median filter to minimize salt and pepper noise
        blur = cv2.medianBlur(img, 5)
        
        # convert image to black (background) and white (worms)
        thres_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, constant) 
        
        # sequentially apply masks to each image                                                                       
        roi_ctrl = thres_img & mask_ctrl                                                                     
        roi_exp = thres_img & mask_exp 
        
        # count number of white (worm) pixels in ctrl ROI
        count_ctrl = 0                                                          
        for i in range(Q1y-(rad+10), Q4y+(rad+10)):          
            for j in range(Q1x-(rad+10), Q4x+(rad+10)):
                if roi_ctrl[i][j] == 255:
                    count_ctrl += 1
                else:
                    count_ctrl += 0
        
        # count number of white (worm) pixels in exp ROI
        count_exp = 0                                                           
        for i in range(Q2y-(rad+10), Q3y+(rad+10)):          
            for j in range(Q3x-(rad+10), Q2x+(rad+10)):
                if roi_exp[i][j] == 255:
                    count_exp += 1
                else:
                    count_exp += 0
        
        # append worm pixels in each ROI per frame to empty lists
        pix_count_ctrl.append(count_ctrl)                                      
        pix_count_exp.append(count_exp)
        
        print(count)                                                            
        count += 1
    
    # calculate percentage occupancy in both ROIs for each frame
    occ_ctrl = [(a/count_mask_ctrl)*100 for a in pix_count_ctrl]                
    occ_exp = [(b/count_mask_exp)*100 for b in pix_count_exp]

    # calculate preference index (PI) for each video frame
    # if index equation denominator is zero, indicate NaN
    pi_frame = []                                     
    for i in range (len(occ_ctrl)):
        if occ_exp[i] + occ_ctrl[i] != 0:
            pi_frame.append(((occ_exp[i] - occ_ctrl[i]) / 
                            (occ_exp[i] + occ_ctrl[i])))
        else:
            pi_frame.append(np.nan)
            
    pi_vid = [np.nanmean(pi_frame)]
           
    ## Generate csv output file                                    
    # generate dataframes                              
    df1 = pd.DataFrame(pix_count_ctrl, columns = ['Worm Pixels in Ctrl'])
    df2 = pd.DataFrame(pix_count_exp, columns = ['Worm Pixels in Exp'])
    df3 = pd.DataFrame(occ_ctrl, columns = ['% Occupancy in Crtl'])      
    df4 = pd.DataFrame(occ_exp, columns = ['% Occupancy in Exp'])
    df5 = pd.DataFrame(pi_frame, columns = ['PI per Frame'])
    df6 = pd.DataFrame(pi_vid, columns = ['PI Video'])
    
    # concatenate dfs into result datasheet
    frames = [df1, df2, df3, df4, df5, df6]                               
    result = pd.concat(frames, axis=1)
    result.to_csv(csvfilename)
    
    ## Generate worm occupancy figure
    # concatenate necessary lists and pivot 
    pix = pix_count_ctrl + pix_count_exp
    treatment = ['ctrl']*count + ['exp']*count
    time = list(range(count)) + list(range(count))
    df0 = pd.DataFrame(time, columns = ['Time'])
    df1 = pd.DataFrame(pix, columns = ['Pix'])
    df2 = pd.DataFrame(treatment, columns = ['Treatment'])
    frames = [df0, df1, df2]                  
    heatplot_df = pd.concat(frames, axis=1)
    heatplot_df = heatplot_df.pivot("Treatment", "Time", "Pix")
    max_pix = max(pix) 
    
    # generate and save plot
    fig, ax = plt.subplots()
    ax = sns.heatmap(heatplot_df, cbar_kws = {"orientation": "horizontal"}, 
                     vmin = 0, vmax = max_pix)
    
    ylabels = ['Exp','Ctrl']
    ax.set_yticklabels(ylabels)
    ax.tick_params(axis='y', labelsize=15)
    
    ax.xaxis.set_ticks(np.arange(0, count, 60))
    xlabels = [str(x//60) for x in range(0, count+1, 60)]
    ax.set_xticklabels(xlabels)
    
    plt.ylabel('ROI', fontsize = 14, fontweight = 'bold')
    plt.xlabel('Time (min)', fontsize = 14, fontweight = 'bold')
    plt.title('PI = %s'%(pi_vid[0]), fontsize = 18, fontweight = 'bold')
    
    fig.savefig(pngplot)
    plt.close(fig)

    return;