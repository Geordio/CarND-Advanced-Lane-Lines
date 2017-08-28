## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This Readme forms the report for my submission of the Udacity Advance Lane Finding project.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

My Project Submission
---
This readme file provides the report writeup for this project.

My project solution consists of the following files


| File        | Description        |
| ------------- |:-------------:|
| calibration.py      | handles the camera calibration and undistortion |
| colourfiltering.py | handles the colour filtering to create binary images, and alos the gradient thresholding     |
| Frame.py | The Frame class cordinates the processing of images other than colour thresholding. Frame is also responsible cooridinating the undistortion, extraction oof pixel representations of lines, and creating output images      |
|Line.py | the Line class stores parameters relavent to a Lane Line, and also checks line plausibility, averages line quadratic representation and calculates radius
|pipeline.py| defines the main pipeline sequence|
|util.py| provides some helper functions such as to plot figures|


## Introduction

In my project, I actually swapped the color transform and perspective transform. I.e I performed the perspective transform first, then the colour transform. The reason for this was that was that this gives a colour image with the road marking close to vertical, and is better for using the a histogram to measure the 'noise' when performing the colour transforms.

In hindsight I spent to much time and effort trying to make my solution object oriented, which resulted in duplication or variables, lots of refactoring, but little benefit.

## Camera Calibration

The first step of my pipeline is to calibrate the camera.
My implementation was based upon the example in the course material.

The method iterates though the provided calibration images, finding the chessboard corners ( expecting 9 x 6 interal points where 2 black squares intercept). This give me an array of points that can be used for calibration.
After finding all the corners, I drew them to visualise.
Below are 2 samples

![corners found](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/corners_found1.jpg)
![corners found](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/corners_found2.jpg)

I then use the open cv calibrateCamera method to return the parameters, Distortion coefficiengs and camera matrix, that I need to undistort images.

See initial raw, distorted image shown below

![distorted image](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/calibration1.jpg)

and the subsequent undistorted image shown below
![undistorted image](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/test_undist.jpg)

I saved the calibration parameters so that they can be retrieved later in my project.


## Distortion Correction of Raw images

Using the Camera Matrix and Distortion coefficents, I undistort the images.
This is done by the method undistort in the calibration.py file.

Below is an example of the raw and undistorted image.
![undistorted image](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/undistorted_driving.png)

As you can see, there are very ubsutle differences that can be picked up around the edges of the image. Particularly near the hood of the vehicle and the tree on the right of the image.


## Birdseye transform

I decided to do the birdseye transform prior to the colour and gradient work. My reason for this is that by doing the transform first, the lines are generally close to vertical, which means that a histogram of pisxel intensities produces a graph with very pronounced peaks.
In addition, the sobel function only needs to be used to look at gradient change on the horizontal axis.

The birdseye transformation is defined in the get_birdseye method in the Frame class.
The method calls calc_warp_points method to return the source and destination warp points. I defined this as a seprate method as the warp points are calculated relative to the image size, and are not defined as absolute values. This is to make the function independant of camera resolution, an issue that I experienced on the initial lane finding project.

Below is an image marked up with the source points (magenta) and the destination points (red)
(Note that the destination points are at the top and bottom of teh image and are quite difficult to see)

![undistorted image](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/src_dst.png)

Below is a sample birdseye transform. Note that this image is of a slight left bend so the lines are not quite vertical.

![undistorted image](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/birdseye_transform.png)

Note that as part of this tranform function, I calculate the inverse transform matrix and store it for use later.



## Apply colour transformations and gradients

I used trial and error to establish the best method for picking out the pixels relavent to the lane lines.

I tried the following colour spaces:

-RBG
-HLS
-HSV
-LAB
-YUV

I split the images into their individual layers and applied independant threseholds to each image.
I tried it against the test_images.
I found that in general, if I tuned the thresholds to do a really good extract of the line information on one test image, that it would perform poorly on another.
The worst cuplrit for this was the test2 image, where the road surface is a different colour.

Thresholding based on the S of the HLS colour space performed better than others, however, I had issues in dealing with areas of shadow, and the areas of lighter stretches of the road
I felt that I needed another solution, I did some research and came across the following article:
http://aishack.in/tutorials/tracking-colored-objects-opencv/

I implemented a method based on this,  'filter_white_yellow_hsv', which uses the open cv method inRange to filter white and yellow colour ranges of the HSV space.

I found that this method had problems coping with the sections of the project_video where the road color changes.

To try to overcome this, I looked at alternative methods, including filtering on the white and yellow colour ranges of the HSV space.
This is defined in colorfiltering.py as filter_white_yellow_hsv.
This method uses the opencv
This worked well on most sections of road, but struggled with areas of shade. In order to attempt to overcome the issue of shade and sections where the road surface is lighter, I implement 2 sets of thresgolds, and swithc between the 2 is the S channel has a mean value of greater or less than 120.
i.e
    if v_mean(img) < 120:
        upper_thres_yell = np.uint8([10,50,100])
        lower_thres_yell = np.uint8([100,255,255])
        upper_thres_white = np.uint8([[0, 0, 200]])
        lower_thres_white = np.uint8([255, 255, 255])
    else:
        upper_thres_yell = np.uint8([10,50,220])
        lower_thres_yell = np.uint8([100,255,255])
        upper_thres_white = np.uint8([[0, 0, 200]])
        lower_thres_white = np.uint8([255, 255, 255])

Note, that to compare the performance, I produced a debug output video that shows the performance, this can be viewed in project_video_debug.mp4
In addition, my experimental methods 'coloumap_sandbox', 'filter_hls' and 'filter_white_yellow_hls2' are defiend in colourfiltering.py

Regarding the gradient of the pixel intensities, I experimented with Sobel, but found that it was not a useful addition to the colour filtering.
Below is an example if the output of the sobel function. The sobel function is defined in colourfiltering as the abs_sobel_thresh method which called by get_combined_binary()


Ulitmately I decided against using the sobel as I found that on some sections (such as the test1 image, it created additional noise that caused spurious results. See output from test.jpg image below
Sobel noise:

![undistorted image](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/combined sobel noise.png)

And the subsequent output
![undistorted image](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/test1_sobel_going_wrong.jpg)


## Detect lane pixels

In my Frame class, the 'get_raw_line_pixels' method includes the function calls to return the pixels associated with each line.
This is done via 2 methods,'find_lanes_near' and 'find_lanes_blind'
Both methods use the sliding window approach to finding lane pixels.
'find_lanes_blind' first ananlyses a histogram of the binary image created by the colour filtering (and sobel). The peaks in each half of the histogram are used as the starting point for the sliding window search.
Any non zero pixels found within the window are treated as being part of the line, the next window up is then drawn, with its centre around the mean centre of the previously found pixles.
This is based on the code from the lesson as I found that it worked out of the box.
An example of the output of this step is found below.

Histogram of the found pixels, with the peaks giving the starting points for the search

![undistorted image](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/histogram.png)

Not that the righthand peak is much lower than the lefthand peak, which is down to the number of pixels being less, as the line is dashed.

Birdseye view with the windows overlaid.
![undistorted image](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/window.png)

If a polyline has previously been fit, 'find_lanes_near' method is used instead. This method uses the previous polyline as a starting point, using the constant value as the basis of a search area.

Note, that at the start of the pipeline, a Line object was created to represent each Line, i.e left and right.
For the output of the 'get_raw_line_pixels' method, the result is passed to the 'fit_poly_lines' method of each line.
This method uses the numpy polyfit method to create a line of best fit the the pixels.

After this the 'addline' method is called, which calls 'check_plausible_line' performs some checking of the plausibility of the caluclated line:

1. checks agains some basic rules about the values of the following thresholds for the quadratic, linear and constant values of the polyline:
        quadratic_high_thresh = 1
        quadratic_low_thresh = 0
        linear_high_thresh = 10
        linear_low_thresh = 0
        const_high_thresh = 2000
        const_low_thresh = -1000

   These are fairly arbitary values that I decided upon by evaluating some random sections of the road. They should mean that some implausible polyfits are discarded, such as overly curved lines, or horzontal lines.

2. checks the difference against the previous average line of best fit. This time the tresholds are:
        quadratic_diff_thresh = 0.005
        linear_diff_thresh = 0.5
        const_diff_thresh = 300

   this checks that the change in the line position between 2 lines should not be significant.


Foloowing this, the line information is added to an array of most recent lines and teh average taken. (at time of writing the average is taken over 5 lines)
This ellimninates a lot of the spurious lines that can be detected, especialy on the dahsed line markings.

## Radius and Vehicle position

### Line radius detection

The Line class is responsible fro detecting the radius of a single line.


The Frame class is then responsible for averaging the radius of the left and right lines.

In the Line class, the 'calculate_radius_real' method calculaes the real world radius. It does this by rescaling the pixel representation of the line, so that they are scaled to match the real world size.
I determined the metre per pixel values by measuring the line geometry on a birseye image, with the distance between the lines being 3.7 metres, and the length of a dashed line being 3 mtres long.
This gave me:
        self.ym_per_pix = 3. / 160  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 560  # meteres per pixel in x dimension

Once the lanes are scaled appropriately, the polyline method is again used to calculate a quadratic representation.


From the website: http://www.intmath.com/applications-differentiation/8-radius-curvature.php
The formula below is used to calculate the radius:

![undistorted image](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_images/radius.png)

Or as implemented in my calculate_radius_real method:
        rad = ((1 + (2 * fit_cr[0] * y_eval * self.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])


Once the radius is calculated, the Frame class averages the radius of teh left and right lines together, which is then used as the overal radius.

### Vehicle position

The vehicle position was detected using the get_car_position method in the Frame class

This method assumes taht the camera is mounted in the car centre line, hence the centre of the car is the horizontal centre of the image.
The centre of the lane is detected my adding the x positions at of the lines nearest the car together and dividing by 2.
The position of the car relative to the lane centre can be determined by substitution. Note that the distnace is converted from pixels to metres using the metres per pixel value as before.


## Warp the detected lane boundaries back onto the original image.

This is done by the 'create_output_drive_view' method in the frame class.

This method uses the inverse distortion matrix that was calculated during the birdseye transformation.

A blank image is created upon which the polyfit representations of the lines of best fit are drawn. The opencv plyfit method is used to draw a line to the pixels that represent each line.
The opencv fillPoly method is used to fil the area between the pixel representation of each line.
At this stage we have a birdseye representation of the lines. We use the opencv cv2.warpPerspective method, this time with the inverse transformation matrix to map these lines back into the driving perspective.

Finally we merge this line view with the original camera view using the opencv addWeighted method.

* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

At this stage we have a marked up camera image with the lane marking in blue and the lane itself in green. In order to annotate the image with the radius and vehicle position information, the annotate_image method is used in the Frame class.
This method takes upto 3 strings as an input and uses the opencv puttext method to write the data to the image.


## Discussion: Video processing


### General
I jumped into the video processing fairly early on in the project, hence a lot of my progress in my solution came through my learning via the video implemention.

I made my solution too complex by trying to make it object oriented. In hindsight this was a mistake for the scope of the project. Too often I found myself refactoring the soltuion to try to come up with the best class to perform a function, in hindsight I would not have structred it as it is now.

I found that my averaging the quadratic coefficients over 5 frames producted a fairly stable lane output.
I performed some fairly rudimentary plausibility of the lane checking was enough to prevent completely spurious lanes being detected, however there is much more that could be done to improve this.
Currently I handle each line independantly. This was a decsion that I made when implementig the Line class. However, in hindsight I should have implemented a Lane LClass that hold the left and right line objects.
Then the the Lane class I could do some additional plausibility checking, such as:
-checking that the identified lines are around 3.7m apart
-checking that the lines are approximately parallel


In addition, I trialed the concept of messuring the signal to noise ratio of the histograms of the colour filtered binary images. I think I could have progress this further to choose alternative colourspaces to work in depending upon the background noise and how high the peaks were.
My alternative solution of switching between 2 thresholds works adequately for this project
A simple mask could have helped to crop unnecessary information from the filtered binary images would have been an easy addition.



### Limitations
-The current implementation of handling the images by expecting that the lanes should be in the left and right halfs of the image means that it would not be able to handle lane changes and track the new lanes efficently.
-The current implement is unlikely to work well in darkness
-The current implementation will struggle with twisty roads, or roads that have significant inclines and declines
-Implementation hasn't been optimised for performance and currently cant work in real time

My output videos are:
Output of project_video.mp4 [project video output](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_project_video.mp4)
Output of challenge_video.mp4 [challenge video output](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_challenge_video.mp4)

(The additional debug video outputs for each are:
[project video output](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_project_video_debug.mp4)
[challenge video output](https://github.com/Geordio/CarND-Advanced-Lane-Lines/blob/master/output_challenge_video_debug.mp4)
