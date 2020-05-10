Writeup - Advanced Lane Finding Project
=======================================

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

[image1]: ./test_images_output/example-distortion-corrected.png "Undistorted"
[image2]: ./test_images_output/steps_test1.png "Example pipeline test1.png"
[image3]: ./test_images_output/steps_straight_lines1.png "Example pipeline straight_lines_1.png"
[video1]: ./test_videos_output/project_video.mp4 "Video"

# Project overview
```
├── README.md                   # This file
├── adv_lane_finding.ipynb      # Pipeline for images and videos
├── camera_cal                  # Camera calibration images
│   └── *.jpg
├── lib                         # Python modules
│   ├── __init__.py
│   ├── camera_calib.py         # Camera calibration code
│   ├── constants.py            # Various constants that may be used in multiple files
│   ├── lane_pixel_ops.py       # Lane line pixel ops: lane line pixel extraction, perspective transform
│   ├── line.py                 # Lane line, including polynomial fitting
│   └── road_lane.py            # Road lane, combines the left and right lane line
├── test_images
│   └── *.jpg                   # Input test images
├── test_images_output
│   ├── example_distortion-corrected.png 
│   └── steps_*.png             # Processed test images, showing all pipeline steps.
├── test_videos                 # Input videos
│   ├── challenge_video.mp4
│   └── project_video.mp4
└── test_videos_output          # Pipeline applied on videos, showing detected lane + metrics
    ├── challenge_video.mp4
    └── project_video.mp4
```

# Rubric points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `lib/camera_calib.py`. I created a class called `Undistorter` with public methods `cablibrate()` and `apply()`. 

In the calibration step, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

The distortion correction is applied with the `apply()` method, which calls `cv2.undistort()`. Example of a distortion corrected calibration image: 

![alt][image1]

### Pipeline (single images)

Please see these two figures for `test1.jpg` and `straight_lines1.jpg` for a demonstration of all steps in the pipeline:

![alt][image2]
![alt][image3]

Figures for the other test images are available in the `test_images_output` directory.

#### 1. Provide an example of a distortion-corrected image.

See the images with title "Undistorted". The field of view is a bit cropped due to the distortion correction, which can be seen clearly by comparing the tail lights of the white car for `test1.jpg`.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. I first converted the image to HLS color space, after which I applied the Sobel operation
on the L (lightness) channel in the x-direction to calculate gradients. This functions as an edge detector, and the x-direction is better at picking up lane lines (from the front camera perspective) than the y-direction. These gradients, taken absolutely and normalized to range 0 - 255, are then thresholded between 40 and 140. As a result, any value within the threshold is set to 1, while the remainder is set to 0.

The Sobel operation does not work very well in shaded areas, because gradients here are less pronounced. To compensate for this, I also use a simple color threshold on the S (saturation) channel. The S channel picks up lane lines very well by itself (especially yellow) and by settng the threshold relatively high to 155 - 255, I make sure that the lane itself (the gray/black-ish asphalt) is not picked up, while the lane lines are. The result is binarized in the same way as above. The two detectors are combined using an "OR" operation: if the pixel is activated in either detector (or in both), it is also activated in the result.

In both cases, the thresholds were picked empirically by trying different values and visually comparing the results on the provided test images, with the goal of picking up as much as possible of the lane lines, without triggering on other things such as color changes in the pavement and shadows.

For an example, see the images with title "Lane pixel extraction" in the examples above. The corresponding code can be found in `lib/lane_pixel_ops.py`, lines 29 through 47.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform can be found in `lib/lane_pixel_ops.py`, lines 59 through 71. The figure above shows the source and destination as dashed lines on the images with title "Distorted". These coordinates are hardcoded in `lib/constants.py`, lines 23 to 37.

This resulted in the following source points:

```
array([[ 592.,  450.],
       [ 185.,  719.],
       [1130.,  719.],
       [ 687.,  450.]], dtype=float32)
```

And dest points:
```
array([[300.,   0.],
       [300., 719.],
       [980., 719.],
       [980.,   0.]], dtype=float32)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. See images titled "Perspective transform" in the image above - this is especially clear for `straight_lines_1.jpg`.


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Line fitting is done in `lib/line.py`, class `Line`.

Fitting with the sliding window approach is done in classmethod `from_sliding_window()`, lines 58 - 71.

See the yellow line in the figures above on the second row, separately for left and right.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

##### Convert polyomial coefficients to meter

Consider the following polynomial formula, where `A`, `B` and `C` are coefficients found previously in the fitting procedure:

	x = A y^2 + B y + C

Here, `x` and `y` are espressed in pixels.

To convert to meters, we need scale factors `XM_PER_PIX` and `YM_PER_PIX` (see `lib/constants.py`), which define, for the warped space, how many meters per pixel there are in the x direction and the y direction respectively. With these, the polyomial coefficients can be converted:

	A' = A * XM_PER_PIX / YM_PER_PIX^2
	B' = B * XM_PER_PIX / YM_PER_PIX
	C' = C

This happens in lines 176 to 182 in `lib/line.py`.

##### Compute Radius of Curvature
I calculated radius of curvature separately for each lane line first, and then averaged the results. 

Radius of curvature in meters is computed in lines 164 through 170 in `lib/line.py` by using [the standard formula](https://en.wikipedia.org/wiki/Radius_of_curvature#Definition), plugging in the converted polomial coefficients.

The overall radius of curvature is a mean of these two, calculated in `lib/road_lane.py`, lines 23 through 29. The `RoadLane` class defined in this file encapsulates the two lane lines in order to do overarching calculations that involve both lines.

##### Compute relative position of vehicle

The relative position of the vehicle is calculated in the `lib/road_lane.py` file, lines 32 through 47.

I assume that the camera is mounted in the exact middle of the car.

I evaluate the polomials on `y = 719`, yielding x-values for each polonomial (using the original parameters in pixel space). This is the position of the lane line that is closest to the vehicle, which should be the most accurate estimate.

If the car is driving in the middle of the lane, the x coordinate for the middle of the image should agree with the x coordinate exactly in between the two lane lines. The deviation is calculated as follows:

	car_deviation_center = car_pos_x - self.x_middle 

With `car_pos_x` being the x-coordinate in the middle of the two lane lines and`self.x_middle` being the middle of the image.

If the value is positive, the car is a bit too far too the right. If negative, it is a bit too far to the left.

Finally, this value is multiplied with `XM_PER_PIX` in order to report the deviation in meters rather than pixels.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The plotting of the lane line polygon is done in the Notebook, function `visualize_lane_line()`.

In the figures above, see image with title "Visualize lane + metrics".


---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

For the video pipeline, I implemented:

- searching from a prior (see `lib/line.py` lines 147 to 161, and `lib/road_lane.py` lines 49 to 84)
- sanity checks that verify the width of the lane in the top and the bottom (see `lib/road_lane.py` lines 86 to 100)
- A reset after 25 frames if it failes to recover.
- Side-by-side view with the binary warped image.

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
The pipeline seems to work reasonably well on the project video, even without smoothing.
One issue that I was not able to solve, is a large difference in the radius of curvature between the left and right lane lines when the lane lines are supposed to be straight (see result for `straight_lines2.jpg` above). However, during an actual curve, the radius of curvatures do agree.

Regarding smoothing, I did implement it in the previous project. It did not seem to be needed here to get a good result on the project video.

The pipeline does break down on the challenge video, however. There are various reasons for this:

- The barrier in the middle of the road and shadows cause sharp edges.
- Different pieces of asphalt in a single lane, causing an edge to be detected.
- No smoothing.
- The sanity check for the width of the top of the lane is very sensitive: small mistakes are blown up by the perspective transform, leading to large deviations from the expected 3.7 m.

If I were to keep working on this, I would try to improve it by working on the following:

- Better thresholds for the lane pixel extraction (only the test images were used right now, rather than entire videos).
- A smarter way of initializing the sliding window search, rather than by just cutting the histogram in half and finding peaks. Maybe we can use the expected width of the road. Say, we are very confident about the right line but we have two options to choose out of for the left line, we could choose the one that appears to be at the appropriate distance.
- Exponential moving average smoothing on polynomial parameters.
- More robust sanity checks.

In general, I would also look into a deep learning approach. Trying to find optimal thresholds and hyperparameters manually is very time consuming and probably leads to sub optimal results. Algorithmically finding hyperparameters such that errors on predictions of lane lines are minimized (with respect to annotated ground truth) seems like a more scalable approach to me.
