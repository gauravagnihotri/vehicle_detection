###############################################################
'''slide7.py file is final version of code'''
###############################################################
'''This code section imports the modules required'''
################################################################## 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import os
import fnmatch
import cv2
from PIL import Image
###############################################################
''' this code section adds 'non' in filename for non car images'''
###############################################################
#inp = "/home/garyfirestorm/Documents/sdc_data/Vehicle_Detection/data_set/non-vehicles/non-vehicles/GTI/"
#path=os.listdir(inp)
##images = glob.glob('/home/garyfirestorm/Documents/sdc_data/CarND-Vehicle-Detection/data_set_small/**/*.jpeg', recursive=True)
#ii=0
#for filename in path:
#    if fnmatch.fnmatch(filename, '*.png'):
#        #image = cv2.imread(os.getcwd() +'/train_data/vehicles/vehicles/GTI_Left/'+filename)
#        image = cv2.imread(inp + filename)
#        #lane_det=image[:,:,::-1]
#        #cv2.imwrite(os.getcwd() +'/train_data/vehicles/vehicles/GTI_Left/'+'test'+str(ii)+'.png',image)
#        cv2.imwrite(inp +'non'+str(ii)+'.png',image)
#        os.remove(inp + filename)
#        ii+=1
###############################################################
'''This code section makes arrays for cars and non car images'''
###############################################################

#images = glob.glob('/home/garyfirestorm/Documents/sdc_data/Vehicle_Detection/data_set/**/*.png', recursive=True)
##images = glob.glob('/home/garyfirestorm/Documents/sdc_data/Vehicle_Detection/data_set_small/*.jpeg')
#cars = []
#notcars = []
#for image in images:
#    if 'non' in image:
#        notcars.append(image)
#    else:
#        cars.append(image)
#        
## Just for fun choose random car / not-car indices and plot example images   
#car_ind = np.random.randint(0, len(cars))
#notcar_ind = np.random.randint(0, len(notcars))
#    
## Read in car / not-car images
#car_image = mpimg.imread(cars[car_ind])
#notcar_image = mpimg.imread(notcars[notcar_ind])


# Plot the examples
#fig = plt.figure()
#plt.subplot(121)
#plt.imshow(car_image)
#plt.title('Example Car Image')
#plt.subplot(122)
#plt.imshow(notcar_image)
#plt.title('Example Not-car Image')


###############################################################
'''This code section sets parameters for training the SVM'''
################################################################## 
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb #possibly HLS
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block #this used to be 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [[400,656],[400,656],[390,550]] # Min and max in y to search in slide_window()
window_size = [(128, 128),(96, 96),(80,80)]
colorss=[(0,255, 0),(255,255, 0),(255,0, 0)]
x_start_stop = [[412, None],[412, None],[412, 1280]]
overlaps = [(0.5, 0.5),(0.5, 0.5),(0.5, 0.5)]
###############################################################
'''This code section trains the model'''
################################################################## 
#car_features = extract_features(cars, color_space=color_space, 
#                        spatial_size=spatial_size, hist_bins=hist_bins, 
#                        orient=orient, pix_per_cell=pix_per_cell, 
#                        cell_per_block=cell_per_block, 
#                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                        hist_feat=hist_feat, hog_feat=hog_feat)
#notcar_features = extract_features(notcars, color_space=color_space, 
#                        spatial_size=spatial_size, hist_bins=hist_bins, 
#                        orient=orient, pix_per_cell=pix_per_cell, 
#                        cell_per_block=cell_per_block, 
#                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                        hist_feat=hist_feat, hog_feat=hog_feat)
#
#X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
## Fit a per-column scaler
#X_scaler = StandardScaler().fit(X)
## Apply the scaler to X
#scaled_X = X_scaler.transform(X)
## Define the labels vector
#y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
## Split up data into randomized training and test sets
#rand_state = np.random.randint(0, 100)
#X_train, X_test, y_train, y_test = train_test_split(
#    scaled_X, y, test_size=0.2, random_state=rand_state)
#
#print('Using:',orient,'orientations',pix_per_cell,
#    'pixels per cell and', cell_per_block,'cells per block')
#print('Feature vector length:', len(X_train[0]))
## Use a linear SVC 
#svc = LinearSVC(C=0.01)
## Check the training time for the SVC
#t=time.time()
#svc.fit(X_train, y_train)
#t2 = time.time()
#print(round(t2-t, 2), 'Seconds to train SVC...')
## Check the score of the SVC
#print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
## Check the prediction time for a single sample
#t=time.time()
###############################################################
'''This code section stores the model in a pickle file'''
################################################################## 
#
#with open(os.getcwd() + '/classifier.p','wb') as f:
#    pickle.dump(svc,f)
#    pickle.dump(X_scaler,f)

###############################################################
'''This code section defines some functions that are used'''
################################################################## 

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
    # Return the image
    return img
###############################################################
'''This code section loads the classifier pickle file'''
################################################################## 
with open(os.getcwd() + '/classifier.p','rb') as f:
    svc = pickle.load(f)
    X_scaler = pickle.load(f)
####################################################

###################################################
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

class BoxQueue():
    def __init__ (self):
        self.queue_len = 10
        self.new_boxes = []

    def add_hot_box (self, boxes):
        if (len(self.new_boxes) > self.queue_len):
            tmp = self.new_boxes.pop (0)
        
        self.new_boxes.append (boxes)
        
    def return_hot_box (self):
        b = []
        for boxes in self.new_boxes:
            b.extend (boxes)
        return b

total_hot_boxes = BoxQueue()

###############################################################
'''This code section defines pipeline used for performing sliding search, heatmapping and applying averaging'''
################################################################## 
def detect_veh(fname):
    draw_image = np.copy(fname)
    heat = np.zeros_like(draw_image[:,:,0]).astype(np.float)
    heat1 = np.zeros_like(draw_image).astype(np.float)
    heatout = np.zeros_like(draw_image).astype(np.float)
    fname = fname.astype(np.float32)/255
    all_hot_windows = []
    for ii in range(0,len(y_start_stop)):
        windows = slide_window(fname, x_start_stop=x_start_stop[ii], y_start_stop=y_start_stop[ii], 
                        xy_window=window_size[ii], xy_overlap=overlaps[ii])
        hot_windows = search_windows(fname, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        all_hot_windows.extend(hot_windows)
    total_hot_boxes.add_hot_box(all_hot_windows) 
    all_hot_windows = total_hot_boxes.return_hot_box()                      
    heat = add_heat(heat,all_hot_windows)
    heat = apply_threshold(heat,2)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    draw_image = draw_labeled_bboxes(np.copy(draw_image), labels)
    heat1[:,:,0] = add_heat(heat1[:,:,0],all_hot_windows)
    heat1[:,:,1] = add_heat(heat1[:,:,1],all_hot_windows)
    heat1[:,:,2] = add_heat(heat1[:,:,2],all_hot_windows)
    heat1 /= (heat1.max()/255.0)
    heat1 = heat1.astype(np.uint8)
    heatout = cv2.applyColorMap(heat1, cv2.COLORMAP_JET)
    draw_image = np.vstack((heatout[:,:,::-1],draw_image))
    return draw_image

###################################################################
#video proc
###################################################################
from moviepy.editor import VideoFileClip
from IPython.display import HTML
name = 'project'
white_output = name + '_video_processed.mp4'
clip1 = VideoFileClip(name + "_video.mp4")
white_clip = clip1.fl_image(detect_veh) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

##################################################################
#image proc
###################################################################
#images = glob.glob('/home/garyfirestorm/Documents/sdc_data/Vehicle_Detection/test_images/*.jpg', recursive=True)
#ii=0
#for image in images:
#    img = mpimg.imread(image) 
#    out_img = detect_veh(img)
#    #cv2.imwrite(os.getcwd() +'/dump_proc/'+'0.jpeg',out_img[:,:,::-1])
#    cv2.imwrite(os.getcwd() +'/dump_proc/'+'test_' + str(ii)+'.jpeg',out_img[:,:,::-1])
#    ii+=1
##    fig = plt.figure(figsize=(16,10))
##    plt.imshow(out_img)