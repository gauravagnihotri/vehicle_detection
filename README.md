**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG_test.png "HOG Implementation on car image"
[image2]: ./output_images/non_carHOG_test.png "HOG Implementation on non-car image"
[image3]: ./output_images/window1.jpeg "Search Window Area 1"
[image4]: ./output_images/window2.jpeg "Search Window Area 2"
[image5]: ./output_images/window3.jpeg "Search Window Area 3"
[image6]: ./output_images/all_windows.jpeg "Search Window Area - All"
[image7]: ./test_images/test1.jpg "Test Image 1"
[image8]: ./output_images/test_3_proc.jpeg "Test Image 1 Processed"
[image9]: ./test_images/test6.jpg "Test Image 2"
[image10]: ./output_images/test_1_proc.jpeg "Test Image 2 Processed"
[image11]: ./test_images/test2.jpg "Test Image 3"
[image12]: ./output_images/test_5_proc.jpeg "Test Image 3 Processed"

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Lines 75 through 84 (in vehicle_detect.py) defines the parameters used to extract the hog features. The ```YCrCb``` colorspace was used with 'ALL' Channels.
Before performing HOG feature extraction, spatial binning feature extraction was performed with spatial size of ```(16,16)```, Histogram Feature Extraction was performed with ```32``` histogram bins. The ```extract_features``` function was used to extract features of cars and non-car images. The ```extract_features``` is used from the ```lesson_functions.py```[1] The ```get_hog_features``` function was used to extract the HOG features (lines 6 through 23 in ```lesson_functions.py```

![alt text][image1]
*Figure shows each channel (of YCrCb colorspace)  for a sample car image along with their HOG implementations*
![alt text][image2]
*Figure shows each channel (of YCrCb colorspace) for a sample non-car image along with their HOG implementations*

After exploring HLS, HSV and LUV colorspaces, I found out that YCrCb worked better in identifying the car images. The detection rate and number of duplicates was optimal with YCrCb colorspace. Multiple detections are important, since they help in removing false detections. 

#### 2. Explain how you settled on your final choice of HOG parameters.

```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb #possibly HLS

orient = 9  # HOG orientations

pix_per_cell = 8 # HOG pixels per cell

cell_per_block = 2 # HOG cells per block

hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
```

`Using: 9 orientations 8 pixels per cell and 2 cells per block

Feature vector length: 6156`

Based on trial and error in detecting vehicle in a test image, I used `9` for HOG orientations, `8 HOG pixels per cell` and `2 HOG cells per block`. I observed slight improvement in accuracy with using `ALL` channels of the image in YCrCb colorspace

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Lines 93 through 131 (in vehicle_detect.py) is the code section used for training the LinearSVC (with linear kernel) ```svc = LinearSVC(C=0.01)```
```C=0.01``` regularisation parameter was used. Lower C value should cause some misclassification, but the hyperplane which is drawn between the separate datasets is more closer to each dataset. The ```features``` matrix with color features, and HOG features may cause the datasets to be very close, hence lower C value was used here. Experimentally, the lower C value resulted in better classification. ```C = 1, 0.1 and 0.01``` was tried. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Functions ```slide_window``` (lines 103 through 142 in lesson_functions.py) and ```search_windows``` (lines 231 through 259 in vehicle_detect.py) were used in performing sliding window search (lines 299 through 306 in lesson_functions.py) on a selected section of the video.
The following parameters were used to perform the sliding search. ```x_start_stop``` and ```y_start_stop``` are X and Y limits used for performing the search. 
Window sizes used for these arrays were `128,128`, `96,96` and `80,80` with 50% overlap for each window.
```
y_start_stop = [[400,656],[400,656],[390,550]] # Min and max in y to search in slide_window()

window_size = [(128, 128),(96, 96),(80,80)]

x_start_stop = [[412, None],[412, None],[412, 1280]]

overlaps = [(0.5, 0.5),(0.5, 0.5),(0.5, 0.5)]
```

| Window Search Area 1 | Window Search Area 2 |
|:---:|:---:|
| ![alt text][image3] | ![alt text][image4] |
| Window Search Area 3 | Window Search Area - All |
| ![alt text][image5] | ![alt text][image6] |

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

| Test Image 1 | Test Image 1 Processed |
|:---:|:---:|
| ![alt text][image7] | ![alt text][image8] |
| Test Image 2 | Test Image 2 Processed |
| ![alt text][image9] | ![alt text][image10] |
| Test Image 3 | Test Image 3 Processed |
| ![alt text][image11] | ![alt text][image12] |
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

| Original Project Video | Processed Project Video |
|:---:|:---:|
| [![Vehicle Detection Raw](https://i.ytimg.com/vi/ntsQ03OSk7s/maxresdefault.jpg)](https://youtu.be/ntsQ03OSk7s) | [![Vehicle Detection Processed](https://i.ytimg.com/vi/l7zqSn8HCXg/maxresdefault.jpg)](https://youtu.be/l7zqSn8HCXg) |


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

### References

