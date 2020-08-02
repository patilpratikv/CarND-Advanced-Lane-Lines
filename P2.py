# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 09:03:20 2020

@author: pratik patil
"""


#%% 

# Importing required libraries

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip

#%%

class Line:
    """
        Define a class to receive the characteristics of each line detection

    """
    def __init__(self):   
        """
            Initialization for the class

        """
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        
    def best_guess_from_prev_frames(self, poly_coeff, indx):
        # add a found fit to the line, up to n
        if poly_coeff is not None:
            if self.best_fit is not None:
                # Coefficient difference with current and old
                self.diffs = abs(poly_coeff - self.best_fit)
            if (self.diffs[0] > 0.001 or \
               self.diffs[1] > 1.0 or \
               self.diffs[2] > 100.) and \
               len(self.current_fit) > 0:
                self.detected = False
            else:
                self.detected = True
                self.px_count = np.count_nonzero(indx)
                self.current_fit.append(poly_coeff)
                if len(self.current_fit) > 5:
                    # consider only last 5 frames
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # If good guess didn't found remove it from history
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out last 
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any guesses from past in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)


#%% 

def getObjImgPoints(images, nX, nY):
    """  
    Function to get object and image points from calibration images.
    
    Parameters
    ----------
    images : Array 
        Images filename in calibration folder.
    nX : Int
        Number of internal corners on chess board in image along x axis.
    nY : Int
        Number of internal corners on chess board in image along y axis.

    Returns
    -------
    objPoints : Array
        Object points for the image
    imgPoints : Array
        Corner points in the image
    imageShape: List
        Image diemnsions
    """
    
    objPoints = []
    imgPoints = []
    
    objP = np.zeros((nY * nX, 3), np.float32) # Object points init
    objP[:,:2] = np.mgrid[0:nX, 0:nY].T.reshape(-1,2) # X,Y Co-ordinate
    
    # Loop over all images 
    for imagefileName in images:
        
        # read image
        image = mpimg.imread(imagefileName)
            
        # convert image to grayscale
        grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # find chessboard corner in image
        status, corners = cv2.findChessboardCorners(grayImage, (nX, nY), None)
        #if the 'status' is true from previous function update image and object points
        if status == True:
            imgPoints.append(corners)
            objPoints.append(objP)
            # draw points on image
            image = cv2.drawChessboardCorners(image, (nX, nY), corners, status)
            
    return objPoints, imgPoints


#%%
    
def caliberateCamera():
    """    
    Function to calibrate camera using chessboard images.
    
    Returns
    -------
    objPoints : Array
        Object points for the image
    imgPoints : Array
        Corner points in the image

    """
    # Read all images from camera_cal folder    
    cal_images = glob.glob('camera_cal/calibration*.jpg')
    
    nX = 9
    nY = 6    
    
    objPoints, imgPoints = getObjImgPoints(cal_images, nX, nY)

    return objPoints, imgPoints
    
    
#%%
    
def undistortImg(img, objPoints, imgPoints):
    """    
    Function to get undistorted image.
    
    Parameters
    ----------
    img : Array 
        Images filename to undistort.
    objPoints : Matrix
        Object Points from chessboard image.
    imgPoints : Matrix
        Image Points of corners from chessboard image.
        
    Returns
    -------
    undist : Image matrix
        Undistorted image

    """
    #test_image = mpimg.imread(img)
    test_image = np.copy(img)
    image_size = (test_image.shape[1], test_image.shape[0])
    
    # calibrate camera matrix values
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, image_size, None, None)
    # undistort original image
    undist = cv2.undistort(test_image, mtx, dist, None, mtx)
    
    return undist
    

#%%
    
def unwarp(img):
    """    
    Function to unwarp image.
    
    Parameters
    ----------
    img : Matrix 
        Images matrix.
        
    Returns
    -------
    warped : Image matrix
        Warped image matrix
    M : Matrix
        Perspective transform matrix

    """
    height, width = img.shape[0], img.shape[1]
    
    # define source and destination points for transformation
    srcPoints = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])
    dstPoints = np.float32([(450,0),
                  (width - 450,0),
                  (450, height),
                  (width - 450, height)])
    
    M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    Minv = cv2.getPerspectiveTransform(dstPoints, srcPoints)
    warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_LINEAR)
    
    return warped, M, Minv


#%%
    
def HLS_L_channel(img, thresh=(220, 255)):    
    """    
    Function that thresholds the L-channel of HLS color space.
    For white lines in image.
    
    Parameters
    ----------
    img : Array 
        Image array
    thresh : Tuple
        Tuple with Lower and uppar threshold for the L channel. (lower, uppar)
        
    Returns
    -------
    lChannel_thresh : Image matrix
        Binary image with required line detection with threshold
        
    """
    # Convert to HLS 
    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lChannel = HLS[:,:,1]
    lChannel = lChannel*(255/np.max(lChannel)) # normalize
    # Apply threshold
    lChannel_thresh = np.zeros_like(lChannel)
    lChannel_thresh[(lChannel > thresh[0]) & (lChannel <= thresh[1])] = 1
    
    return lChannel_thresh


#%%
    
def LAB_B_channel(img, thresh=(190, 255)):    
    """    
    Function that thresholds the B-channel of LAB color space.
    For yellow lines in image.
    
    Parameters
    ----------
    img : Array 
        Image array
    thresh : Tuple
        Tuple with Lower and uppar threshold for the B channel. (lower, uppar)
        
    Returns
    -------
    bChannel_thresh : Image matrix
        Binary image with required line detection with threshold
        
    """
    # Convert to HLS 
    LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    bChannel = LAB[:,:,2]
    if np.max(bChannel) > 175: # if there are yellow lines
        bChannel = bChannel*(255/np.max(bChannel)) # normalize
    # Apply threshold
    bChannel_thresh = np.zeros_like(bChannel)
    bChannel_thresh[(bChannel > thresh[0]) & (bChannel <= thresh[1])] = 1
    
    return bChannel_thresh


#%%
    
def lane_edge_detection(img, objPoints, imgPoints):    
    """    
    Function to detect lane eddges in the image
    
    Parameters
    ----------
    img : Matrix 
        Image matrix
    objPoints : Matrix
        Object Points from chessboard image.
    imgPoints : Matrix
        Image Points of corners from chessboard image.
        
    Returns
    -------
    edge_img : Matrix
        Binary image with lanes in image
        
    """
    
    # undistort image
    undist_img = undistortImg(img, objPoints, imgPoints)
    # unwarp undistorted image
    unwarp_img, M, Minv = unwarp(undist_img)
    #plt.imshow(unwarp_img)
    # extract edges using l channel in HSV color space
    lChannel_img = HLS_L_channel(unwarp_img)
    # extract edges using b channel in LAB color space
    bChannel_img = LAB_B_channel(unwarp_img)
    # Combine l and b channel thresholds
    edge_img = np.zeros_like(lChannel_img)
    edge_img[(lChannel_img == 1) | (bChannel_img == 1)] = 1
    
    return edge_img, Minv


#%%
    
def sliding_window(img):
    """
    Function to fit polynomial to extracted edges using sliding widow approach

    Parameters
    ----------
    img : Matrix
        Image Matrix

    Returns
    -------
    out_img: Matrix
        Image with sliding window and detected lines within window.
    left_fit: Array
        Coefficient for left line lane
    right_fit: Array
        Coefficient for right line lane
        
    """
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis = 0)
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    # one fourth along x axis
    oneFourthpoint = np.int(midpoint//2)
    # left and right base starting points
    leftx_base = np.argmax(histogram[oneFourthpoint:midpoint]) + oneFourthpoint
    rightx_base = np.argmax(histogram[midpoint:(midpoint+oneFourthpoint)]) + midpoint
    
    ## Visualization ##
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img, img, img))
    
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 12
    # Set the width of the windows +/- margin
    margin = 75
    # Set minimum number of pixels found to recenter window
    minpix = 35
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(img.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        ## Visualization ##
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each using `np.polyfit`
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)
    
    ## Visualization ##
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, left_lane_inds, right_lane_inds


#%%
    
def search_around_poly(img, left_fit_prev, right_fit_prev):
    """
    Function to find lane lines in current video frame based on lane lines in previous line.    

    Parameters
    ----------
    img : Matrix
        Image matrix.
    left_fit_prev: Array
        Plynomial coefficients for left lane from previous frame in video.        
    left_fit_prev: Array
        Plynomial coefficients for right lane from previous frame in video.

    Returns
    -------
    left_fit_new: Array
        New plynomial coefficients for left lane.
    right_fit_new: Array
        New plynomial coefficients for right lane.

    """
    
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 80
    left_lane_inds = ((nonzerox > (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] - margin)) & 
                      (nonzerox < (left_fit_prev[0]*(nonzeroy**2) + left_fit_prev[1]*nonzeroy + left_fit_prev[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] - margin)) & 
                       (nonzerox < (right_fit_prev[0]*(nonzeroy**2) + right_fit_prev[1]*nonzeroy + right_fit_prev[2] + margin)))  

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    left_fit_new, right_fit_new = (None, None)
    if len(leftx) != 0:
        # Fit a second order polynomial
        left_fit_new = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit_new = np.polyfit(righty, rightx, 2)
        
    return left_fit_new, right_fit_new, left_lane_inds, right_lane_inds


#%%

def calc_curv_rad(img, left_fit, right_fit, left_lane_indx, right_lane_indx):
    """
    # Function to calculate curvature radius

    Parameters
    ----------
    img : Matrix
        Binary image amtrix
    left_fit : Array
        Left lane polynomial coeeficients.
    right_fit : Array
        Right lane polynomial coeeficients..
    left_lane_indx : Array
        Left lane nonzero indices.
    right_lane_indx : Array
        Right lane nonzero indices..

    Returns
    -------
    left_curverad : Float
        Left curvature radius.
    right_curverad : Float
        Right curvature radius.

    """
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3.048/100 # meters per pixel in y dimension
    xm_per_pix = 3.7/378 # meters per pixel in x dimension
    
    # Define y-value where we want radius of curvature
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
  
    # Identify the x and y positions in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_indx]
    lefty = nonzeroy[left_lane_indx]     
    rightx = nonzerox[right_lane_indx]
    righty = nonzeroy[right_lane_indx]
    
    if len(leftx) != 0 and len(rightx) != 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curve_rad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curve_rad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
      
    return left_curve_rad, right_curve_rad


#%%
    
def draw_lane_original(original_img, edge_img, left_fit, right_fit, Minv, left_curve_rad, right_curve_rad):
    """
    Draw the Detected Lane Back onto the Original Image. Display curvature radius

    Parameters
    ----------
    original_img : Matrix
        Origianl color image.
    edge_img : TYPE
        DESCRIPTION.
    left_fit : Array
        Left lane line coefficients.
    right_fit : Array
        Right lane line coefficients.
    Minv : Matrix
        Inverse perspective transform matrix.

    Returns
    -------
    result: Matrix
        Image with detected lane overlayed on top of original image and curvature radius.

    """
    
    # create copy of image
    new_img = np.copy(original_img)
    
    if left_fit is None or right_fit is None:
        return original_img
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(edge_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    height, width = edge_img.shape
    ploty = np.linspace(0, height - 1, num = height)# to cover same y-range as image
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height)) 
    # Combine the result with the original image
    result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
    
    # radius of curvature
    font = cv2.FONT_HERSHEY_COMPLEX
    text = 'Curvature Radius: ' + '{:04.2f}'.format((left_curve_rad + right_curve_rad)/2) + 'm'
    cv2.putText(result, text, (50, 50), font, 1, (150,200,200), 2, cv2.LINE_AA)
    
    # car position from center
    # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts 
    if right_fit is not None and left_fit is not None:
        car_position = edge_img.shape[1]/2
        l_fit_x_int = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
        r_fit_x_int = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center_dist = (car_position - lane_center_position) * (3.7/378)
    
    direction = ''
    if center_dist > 0:
        direction = 'right'
    elif center_dist < 0:
        direction = 'left'
        
    abs_center_dist = abs(center_dist)
    text = '{:04.3f}'.format(abs_center_dist) + 'm ' + direction + ' of center'
    cv2.putText(result, text, (40, 200), font, 1, (0,0,0), 2, cv2.LINE_AA)
    
    return result


#%%
    
def process_frame(img):
    """
    Function to process individual frame from video

    Parameters
    ----------
    img: Matrix
        Image matrix.

    Returns
    -------
    processed_img: Matrix.
        Processed Image matrix.

    """
    
    new_img = np.copy(img) # create copy
    
    edgelines_img, Minv = lane_edge_detection(new_img, objPoints, imgPoints)
    
    # if left and right lines were detected last frame, use polyfit_using_prev_fit, otherwise use sliding window
    if not left_line.detected or not right_line.detected:
        _, left_fit, right_fit, left_lane_indx, right_lane_indx = sliding_window(edgelines_img)
    else:
        left_fit, right_fit, left_lane_indx, right_lane_indx = search_around_poly(edgelines_img, left_line.best_fit, right_line.best_fit)
        
    # invalidate both fits if x-intercepts of left lane is around 350px (+/- 100px)
    if left_fit is not None and right_fit is not None:
        # calculate x-intercept (bottom of image, x=image_height) for fits
        height = img.shape[0]
        left_fit_x = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
        right_fit_x = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
        
        x_int_diff = abs(right_fit_x-left_fit_x)
        if abs(350 - x_int_diff) > 100:
            left_fit = None
            right_fit = None
    
    left_line.best_guess_from_prev_frames(left_fit, left_lane_indx)
    right_line.best_guess_from_prev_frames(right_fit, right_lane_indx)
    
    # draw the best guess
    if left_line.best_fit is not None and right_line.best_fit is not None:
        left_curve_rad, right_curve_rad = calc_curv_rad(edgelines_img, left_fit, right_fit, left_lane_indx, right_lane_indx)
        processed_img = draw_lane_original(new_img, edgelines_img, left_fit, right_fit, Minv, left_curve_rad, right_curve_rad)
    else:
        processed_img = new_img
        
    return processed_img
#%%
    
# main function

objPoints, imgPoints = caliberateCamera()    

left_line = Line()
right_line = Line()

video_out = 'project_video_output.mp4'
video_in = VideoFileClip('project_video.mp4')

processed_video = video_in.fl_image(process_frame)

processed_video.write_videofile(video_out, audio=False)

# # Read all images from test_images folder    
# test_images = glob.glob('test_images/test*.jpg')

# fig = plt.figure(figsize=(10, 20))
# imageCounter = 1

# for test_image in test_images:
#     fig.add_subplot(8, 2, imageCounter)
#     img = mpimg.imread(test_image)
#     plt.imshow(img)
#     imageCounter += 1
#     edgelines_img, Minv = lane_edge_detection(test_image, objPoints, imgPoints)
#     out_img, left_fit, right_fit, left_lane_indx, right_lane_indx = sliding_window(edgelines_img)
#     left_curve_rad, right_curve_rad = calc_curv_rad(img, left_fit, right_fit, left_lane_indx, right_lane_indx)
#     final_res = draw_lane(img, edgelines_img, left_fit, right_fit, Minv, left_curve_rad, right_curve_rad)
#     fig.add_subplot(8, 2, imageCounter)
#     plt.imshow(final_res)
#     imageCounter += 1

#imagePath = 'test_images/test5.jpg'
#test_image = mpimg.imread(imagePath)
#edgelines_img, Minv = lane_edge_detection(imagePath, objPoints, imgPoints)
#_, left_fit, right_fit, left_lane_indx, right_lane_indx = sliding_window(edgelines_img)
#left_curve_rad, right_curve_rad = calc_curv_rad(edgelines_img, left_fit, right_fit, left_lane_indx, right_lane_indx)
#out_img, left_fit, right_fit, left_lane_inds, right_lane_inds = sliding_window(edgelines_img)
#final_res = draw_lane_original(test_image, edgelines_img, left_fit, right_fit, Minv)
#plt.imshow(final_res)


