# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/intermediate-steps-solidWhiteRight.jpg "solidWhiteRight.jpg"
[image2]: ./test_images_output/intermediate-steps-solidYellowLeft.jpg "solidYellowLeft.jpg"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied Gaussian blur followed by Canny edge detection, which results in a binary image showing edges. The Region-of-Interest mask was applied to only retain edges within the lane the vehicle is driving on. Finally, Hough transform was used to convert the binary map to lines.

In order to draw a single line on the left and right lanes, I modified the `draw_lines()` function by assigning each line found by the Hough transform to either the "left" or "right" category, depending on the angle of the line, and the x-coordinate (with decision boundary being the middle of the image + some padding). The slope of each line is the average of all the slopes within the category. The start coordinates were found by extracting the largest resp. smallest x-coordinate for left resp. right, and in both cases the smallest y-coordinate within the category. The end coordinates are set to the image height for y. To find x, I extrapolate from the start using the slope found earlier. 

Example of the pipeline in action:

![alt text][image1]
![alt text][image2]

After observing that the lines weren't very stable on videos, I realised that lines don't change significantly from frame to frame. Some smoothing would help to improve stability. Hence, I used exponential moving averages on the slopes and the start coodinates. See `class LineDrawer`.

[Click for example video](test_videos_output/solidWhiteRight.mp4).

The optional challenge did not show a great result. However, due to time constraints I opted to skip it.

### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming would be what would happen when driving up an incline or down a decline. The Region of Interest is hardcoded and will not match properly in those scenarios. 

Another shortcoming could be curvy roads. There is an implicit assumption that all lane lines are straight, which is a simplification of reality. Estimating the direction of a curve would be useful data for a self-driving vehicle.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use a dynamic Region of Interest (ROI) mask. Define some function that evaluates the quality of the lane lines (e.g. same length, both ending in apex). If quality is lower than some threshold, try a different hand-made ROI mask, or programatically try to maximize quality by applying small, possibly random, variations on the mask for a fixed number of iterations and keep the best solution. Or maybe even optimize the parameters of the lines mathematically.

Another potential improvement could be to estimate (Bézier?) curves rather than straight lines.

