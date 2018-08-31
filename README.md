## Writeup

**Advanced Lane Finding**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to distorted image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Compute the camera matrix and distortion coefficients. 

The code for this step is contained in the **Distortion Correction* cell of the IPython notebook located in `./Advanced_Lane_Finding_verbose.ipynb` .  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the original chessboard image using the `cv2.undistort()` function and obtained this result: 

![2018061415289629796524.png](http://p37mg8cnp.bkt.clouddn.com/2018061415289629796524.png)

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

I applied distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![20180614152896412260328.png](http://p37mg8cnp.bkt.clouddn.com/20180614152896412260328.png)



#### 2. Perspective transform.

The code for my perspective transform includes a function called `getM_Minv()`, which appears in the 7th code cell of the IPython notebook located in `./Advanced_Lane_Finding_verbose.ipynb` .  The `getM_Minv()` function takes as inputs image size and 5 more parameters. All of these parameters are meant to describe source and destination points. The function return M and Minv.

```python
img_size = (img.shape[1],img.shape[0])
offset = img_size[0]*.2
bot_width = 0.9 # percent of bootom trapizoid height 0.76 0.62
mid_width = 0.125 # percent of middle trapizoid height 0.08 0.043
height_pct = 0.64 # percent for trapizoid height 0.62 0.61
bottom_trim = 1 # percent from top to bottom to avoid car hood 0.935 

def getM_Minv(img_size,bot_width,mid_width,height_pct,bottom_trim,offset):
    src = np.float32([[img_size[0]*(0.5-mid_width/2),img_size[1]*height_pct],  
                      [img_size[0]*(0.5+mid_width/2),img_size[1]*height_pct],  
                      [img_size[0]*(0.5+bot_width/2),img_size[1]*bottom_trim], 
                      [img_size[0]*(0.5-bot_width/2),img_size[1]*bottom_trim]])
    dst = np.float32([[offset, 0],                       # top left
                      [img_size[0]-offset, 0],           # top right
                      [img_size[0]-offset, img_size[1]], # bottom right
                      [offset, img_size[1]]])            # bottom left
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M,Minv
```

This resulted in the following source and destination points:

|  Source  | Destination |
| :------: | :---------: |
| 560,460  |    256,0    |
| 720,460  |   1024,0    |
| 1216,720 |  1024,720   |
|  64,720  |   256,720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` area onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![20180625152989232593963.png](http://p37mg8cnp.bkt.clouddn.com/github/md/warped.png)

#### 3. Create a thresholded binary image using color transforms, gradients.

First, I displayed a lot of image in different color space and found that the following channels is performing well on the test images.

![](http://p37mg8cnp.bkt.clouddn.com/github/md/color.png)

I tried different types of color and gradient thresholds. located in `./Advanced_Lane_Finding_verbose.ipynb` **Display images that processed by different methods**. I found that the gradient thresholds is not performing well  In some cases, and is very slow.

Finally, I dicided to only use color thresholds to generate a binary image.  Here's an example of my output for this step.

![](http://p37mg8cnp.bkt.clouddn.com/github/md/thresh.png)

#### 4.  Identified lane-line pixels and fit their positions with a polynomial

First I take a histogram of the bottom half of the image and find the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines.Then I implement sliding windows on the image to identified lane-line pixels and fit lane lines with a 2nd order polynomial kinda like this:

![](http://p37mg8cnp.bkt.clouddn.com/github/md/slidingwindow.png)

Since the lines don't necessarily move a lot from frame to frame, in the next frame of video I don't need to do a blind search again, but instead I can just search in a margin around the previous line position. 

![](http://p37mg8cnp.bkt.clouddn.com/github/md/skipslid.png)

The green shaded area shows where I searched for the lines this time. So, once I know where the lines are in one frame of video, I can do a highly targeted search for them in the next frame.

**Sanity Check**

To confirm that my detected lane lines are real I checked whether they are separated by approximately the right distance horizontally. If they are, they will be roughly parallel. And I checked that whether the curvature makes sense.

Located in 7th code cell of `./Advanced_Lane_Finding.ipynb`, 

```python
class Tracker(object):
	def fit(self,leftx,lefty,rightx,righty):
```
**Reset**

When my sanity checks reveal that the lane lines I've detected are problematic for some reason, I simply assume it was a bad or difficult frame of video, retain the previous positions from the frame prior and step to the next frame to search again. When I lose the lines for several frames in a row, I start searching from scratch using a histogram and sliding window just like the first step to re-establish my measurement.

**Smoothing**

Even when everything is working, line detections will jump around from frame to frame a bit. Each time I get a new high-confidence measurement, I append it to the list of recent measurements and then take an average over *5* past measurements to obtain a cleaner result.

#### 5.  Radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature ([awesome tutorial here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php)) at any point $x$ of the function $x = f(y)$ is given as follows:
$$
R_{curve}=\frac{[1+(\frac{dx}{dy})^2]^{3/2}}{|\frac{d^2x}{dy^2}|}
$$
In the case of the second order polynomial above, the first and second derivatives are:
$$
f'(y) = \frac{dx}{dy}=2Ay+B\\
f''(y)=\frac{d^2x}{dy^2}=2A
$$
So, equation for radius of curvature becomes:
$$
R_{curve}=\frac{(1+(2Ay+B)^2)^{3/2}}{|2A|}
$$

```python
class Line():
    def cal_radius_of_curvature(self):
        yvals = range(0,720)
        curve_fit = np.polyfit(self.ally*self.ym_per_pix,self.allx*self.xm_per_pix,2)
        self.radius_of_curvature = ((1+(2*curve_fit[0]*yvals[-1]*self.ym_per_pix+curve_fit[1])**2)**1.5)/np.absolute(2*curve_fit[0])
```

The position of the vehicle with respect to center is calculate by the function below.

~~~python
class Tracker(object):
    def cal_offset(self):
        center_diff = (self.right_line.line_base_pos+self.left_line.line_base_pos)/2
        side_pos = 'right' if center_diff >=0 else 'left'
        return center_diff,side_pos
~~~
#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in 7th code cell of `./Advanced_Lane_Finding.ipynb` in the function `map_lane()`.  Here is an example of my result on a test image:

![](http://p37mg8cnp.bkt.clouddn.com/github/md/result.png)

---

### Pipeline (video)

#### final video output. 

videos output are located in `.\output_videos`.

Here are links to my [project video](https://youtu.be/DwlBIhwc2vI), [challenge video](https://youtu.be/hlnZ-9Ghew8) on YouTube for convenience.

---

### Discussion

I tried many color spaces  and various restrictions to deal with the unstable light condition.

The algorithm  can correctly identify lane lines on the project video, challenge video.But the performance of the algorithm on harder challenge video is not good.

Complex environment such as unstable light conditions(Sometimes overexposure),  incomplete lane lines in the harder challenge video is difficult to the algorithm.

More work can be done to make the lane detector more robust, e.g. semantic segmentation to find pixels that are likely to be lane markers (then performing polyfit on only those pixels).