
# Self-Driving Car Engineer Nanodegree Program
## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[im01]: ./output_images/01-calibration.png "Chessboard Calibration"
[im02]: ./output_images/02-undistort_chessboard.png "Undistorted Chessboard"
[im03]: ./output_images/03-undistort.png "Undistorted Dashcam Image"
[im04]: ./output_images/04-unwarp.png "Perspective Transform"
[im05]: ./output_images/05-colorspace_exploration.png "Colorspace Exploration"
[im06]: ./output_images/09-sobel_magnitude_and_direction.png "Sobel Magnitude & Direction"
[im07]: ./output_images/11-hls_l_channel.png "HLS L-Channel"
[im08]: ./output_images/12-lab_b_channel.png "LAB B-Channel"
[im09]: ./output_images/13-pipeline_all_test_images.png "Processing Pipeline for All Test Images"
[im10]: ./output_images/14-sliding_window_polyfit.png "Sliding Window Polyfit"
[im11]: ./output_images/15-sliding_window_histogram.png "Sliding Window Histogram"
[im12]: ./output_images/16-polyfit_from_previous_fit.png "Polyfit Using Previous Fit"
[im13]: ./output_images/17-draw_lane.png "Lane Drawn onto Original Image"
[im14]: ./output_images/18-draw_data.png "Data Drawn onto Original Image"

[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
*Here I will consider the rubric points individually and describe how I addressed each point in my implementation.*

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

`caliberateCamera` function from P2.py is implemented to calibrate the camera using chess board images.

In built OpenCV functions `findChessboardCorners` and `calibrateCamera` are used to get the camera calibration and distortion coefficients. These can then be used by the OpenCV `undistort` function to undo the effects of distortion on any image produced by the same camera. Generally, these coefficients will not change for a given camera (and lens). The below image show a sample image when we draw the identified corners from the chessboard using `drawChessboardCorners`:

![alt text][im01]

The below image show chessboard after calling `undistort`function.

![alt text][im02]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

`undistortImg` function is implemented which takes calibration data points for the camera and correct the distortion in the image. Following is example of the undistorted image. You can the hood is bottom of the image depicts the change.:

![alt text][im03]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I focused on HSL and LAB color spaces and thresholding to identify the lanes in the image. You can find two function `HLS_L_channel` and `LAB_B_channel` in file.

Ultimately, I chose to use just the L channel of the HLS color space to isolate white lines and the B channel of the LAB color space to isolate yellow lines. I did the thresholding for the colors and then normalize it to remove the effect of light in the image. :

![alt text][im04]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In function `unwarp` I did implement the perspective transform of the image. source and destination points are hard coded based on camera position and approximate field of view in which we want to find the lane lines in front of the vehicle.
Following is the example of one of the warped images from test folder.

![alt text][im05]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions `sliding_window` and `search_around_poly`, which identify lane lines and fit a second order polynomial to both right and left lane lines. The first of these computes a histogram of the bottom half of the image and finds the left and right lane lines. Originally these locations were identified from the local maxima of the left and right halves of the histogram. The function then identifies twelve windows from which to identify lane pixels. This effectively "follows" the lane lines up to the top of the binary image. The image below demonstrates how this process works:

![alt text][im06]

The `search_around_poly` function performs basically the same task, but alleviates much difficulty of the search process by leveraging a previous fit (from a previous video frame, for example) and only searching for lane pixels within a certain range of that fit. 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is based upon [this website](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) and calculated in the code cell titled "Radius of Curvature and Distance from Lane Center Calculation" using this line of code (altered for clarity):
```
curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```
In this example, `fit[0]` is the first coefficient (the y-squared coefficient) of the second order polynomial fit, and `fit[1]` is the second (y) coefficient. `y_0` is the y position within the image upon which the curvature calculation is based (the bottom-most y - the position of the car in the image - was chosen). `y_meters_per_pixel` is the factor used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of meters. 

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:
```
lane_center_position = (r_fit_x_int + l_fit_x_int) /2
center_dist = (car_position - lane_center_position) * x_meters_per_pix
```
The car position is the difference between these intercept points and the image midpoint (assuming that the camera is mounted at the center of the vehicle). Both these calculations are done in `calc_curv_rad` and `draw_lane_original` functions in file.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Implemented function `draw_lane_original` to draw the detected line back on original image. Inverse image is projected using inverse perspective matrix `Minv` and overlaid onto the original image. Image below shows an sample from test image:

![alt text][im07]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most of the problems were because of lighting conditions, shadows, discoloration, etc. It was easy to implement the pipeline of project video but i am facing issues to get it work with challenge video actually both of them. Also there will be issue to detect lines if there is white car driving closely to our car it will be very difiicult to identify the lane lines in that scenario or if there is snow on the road. Also I consider if we are not able to identify lane lines we can average over last `n` reading to detect the lane lines.

I've considered a few possible approaches for making my algorithm more robust. Instead of having hard coded image perspective warping we can do it dynamic based on the scene and how long we want to `look ahead` on the road. Also color thresholding can be made dynamic and robust to adjust to different light conditions.
