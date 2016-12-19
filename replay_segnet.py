from __future__ import print_function
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.layers.noise import GaussianNoise
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils

import cv2
import numpy as np
import numpy.ma as ma
import os
import glob
import matplotlib.pyplot as plt


# TEST = '/home/bowen/Desktop/DeepLearning/SegNet/SegImg/cityscapes/512_256/test'       # path to test video folder
IW = 256
IL = 512

void = np.array([0,0,0])            # ego-vehicle, rectification border, out of roi, unlabeled
dynamic = np.array([111,74,0])      # unlabeled moving objects (animal, cart, wheelchair, etc.)
ground = np.array([81,0,81])        # horizontal ground level structure that doesn't fall into any specific ground level class
road = np.array([128,64,128])       # for vehicle legaly
sidewalk = np.array([244,35,232])   # for non-vehicle legaly, including shoulder on highway
building = np.array([70,70,70])     # building
wall = np.array([102,102,156])      # complete block
fence = np.array([190,153,153])     # side block with intervals, including real fences, corn blocks, etc.
bridge = np.array([150,100,100])    # archway
tunnel = np.array([150,120,90])     # tunnel
pole = np.array([153,153,153])      # both pole and group of poles
tlight = np.array([250,170,30])     # traffic light
tsign = np.array([220,220,0])       # traffic sign
vegetation = np.array([107,142,35]) # trees
terrain = np.array([152,251,152])   # all kinds of horizontal vegetation, soil or sand
sky = np.array([70,130,180])        # sky
pedestrian = np.array([220,20,60])  # pedestrian
rider = np.array([255,0,0])         # exposed human riding on mobile objects
car = np.array([0,0,142])           # car
truck = np.array([0,0,70])          # heavy truck
bus = np.array([0,60,100])          # bus
van = np.array([0,0,90])            # van
trailer = np.array([0,0,110])       # trailer
train = np.array([0,80,100])        # train
motorcycle = np.array([0,0,230])    # motorcycle
bicycle = np.array([119,11,32])     # bicycle

URBAN_CLASS = [void, dynamic, ground, road, sidewalk, building, wall, fence, bridge, tunnel, pole, tlight, tsign, vegetation,
               terrain, sky, pedestrian, rider, car, truck, bus, van, trailer, train, motorcycle, bicycle]
URBAN_CLASS = np.array(URBAN_CLASS)

def visualize(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,25):
        r[temp==l]=URBAN_CLASS[l,0]
        g[temp==l]=URBAN_CLASS[l,1]
        b[temp==l]=URBAN_CLASS[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    if plot:
        plt.imshow(rgb)
    else:
        return rgb

# nomalized the original rgb img
def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)
    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]
    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)
    return norm

# lane detection
def img_pipeline_lane(image):
    # gray scale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Gaussian smoothing
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    # Canny edge
    low_threshold = 50
    high_threshold = 210
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    # # polygon region mask
    # mask = np.zeros_like(edges)   
    # ignore_mask_color = 255   
    # imshape = image.shape
    # vertices = np.array([[(0,imshape[1]),(imshape[0]/3, imshape[1]/2), (2*imshape[0]/3, imshape[1]/2), (imshape[0],imshape[1])]], dtype=np.int32)
    # cv2.fillPoly(mask, vertices, ignore_mask_color)
    # masked_edges = cv2.bitwise_and(edges, mask)
    # # Hough Transform
    # rho = 2 # distance resolution in pixels of the Hough grid
    # theta = np.pi/720 # angular resolution in radians of the Hough grid
    # threshold = 10    # minimum number of votes (intersections in Hough grid cell)
    # min_line_length = 2 #minimum number of pixels making up a line
    # max_line_gap = 5    # maximum gap in pixels between connectable line segments
    # line_image = np.copy(image)*0 # creating a blank to draw lines on
    # # Run Hough on edge detected image
    # # Output "lines" is an array containing endpoints of detected line segments
    # lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
    #                             min_line_length, max_line_gap)
    # # Iterate over the output "lines" and draw lines on a blank image
    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         # delete wrong oriented lines by checking k
    #         # if x2 == x1:
    #         #     cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    #         # else:
    #         #     k = float((y2-y1)/(x2-x1))
    #         #     if k > 0.4 or k < -0.4:
    #         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # # Draw the lines on the original image
    # lines_img = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    # # plt.imshow(lines_img)
    # return lines_img

    return edges


# Start model construction
data_shape = IW * IL
model = Sequential()
# model.add(Layer(input_shape=train_data.shape[1:]))
model.add(GaussianNoise(sigma=0.3, input_shape=(3, IW, IL)))
# encoder
model.add(ZeroPadding2D(padding=(1,1), dim_ordering='th'))
model.add(Convolution2D(64,3,3, border_mode='valid', dim_ordering='th'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
model.add(ZeroPadding2D(padding=(1,1), dim_ordering='th'))
model.add(Convolution2D(128,3,3, border_mode='valid', dim_ordering='th'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
model.add(ZeroPadding2D(padding=(1,1), dim_ordering='th'))
model.add(Convolution2D(256,3,3, border_mode='valid', dim_ordering='th'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
model.add(ZeroPadding2D(padding=(1,1), dim_ordering='th'))
model.add(Convolution2D(512,3,3, border_mode='valid', dim_ordering='th'))
model.add(BatchNormalization())
model.add(Activation('relu'))
# decoder
model.add(ZeroPadding2D(padding=(1,1), dim_ordering='th'))
model.add(Convolution2D(512,3,3, border_mode='valid', dim_ordering='th'))
model.add(BatchNormalization())
model.add(UpSampling2D(size=(2,2), dim_ordering='th'))
model.add(ZeroPadding2D(padding=(1,1), dim_ordering='th'))
model.add(Convolution2D(256,3,3, border_mode='valid', dim_ordering='th'))
model.add(BatchNormalization())
model.add(UpSampling2D(size=(2,2), dim_ordering='th'))
model.add(ZeroPadding2D(padding=(1,1), dim_ordering='th'))
model.add(Convolution2D(128,3,3, border_mode='valid', dim_ordering='th'))
model.add(BatchNormalization())
model.add(UpSampling2D(size=(2,2), dim_ordering='th'))
model.add(ZeroPadding2D(padding=(1,1), dim_ordering='th'))
model.add(Convolution2D(64,3,3, border_mode='valid', dim_ordering='th'))
model.add(BatchNormalization())

model.add(Convolution2D(len(URBAN_CLASS), 1, 1, border_mode='valid', dim_ordering='th'))
model.add(Reshape((len(URBAN_CLASS),data_shape), input_shape=(len(URBAN_CLASS), IW, IL)))
model.add(Permute((2, 1)))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer='adadelta',metrics=["accuracy"])
print (model.summary())
model.load_weights('bowen_urbanhunter_ep400.h5')

vidcap = cv2.VideoCapture('/home/bowen/Desktop/DeepLearning/SegNet/SegImg/cityscapes/512_256/test/stuttgart_01.mp4')
# vidcap = cv2.VideoCapture('/home/bowen/Desktop/Camera/cam.mp4')
# vidcap = cv2.VideoCapture('urban.mp4')
while True:
    n, frame = vidcap.read()
    x, y = frame.shape[0], frame.shape[1]
    frame = frame[0:x-200,0:y]
    # print (frame.shape)
    if frame.shape[0] != IW or frame.shape[1] != IL: 
        re_frame = cv2.resize(frame, (IL,IW))
    else:
        re_frame = frame


    image = np.rollaxis(normalized(re_frame),2)

    output = model.predict_proba(np.array([image]))
    pred = visualize(np.argmax(output[0],axis=1).reshape((IW,IL)), False)
    cars = np.argmax(output[0],axis=1).reshape((IW,IL))
    mask = ma.masked_where(cars>=18, cars).mask
    

    re_frame[mask]=[0,10,255]
    re_frame[~mask]=[0,0,0]
    re_frame = img_pipeline_lane(re_frame)
    cv2.imshow('cars', re_frame)
    # cv2.imshow('origin', frame)
    # cv2.imshow('frame', frame)
    # cv2.imshow('cars', cars)


    if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break






# pred = visualize(np.argmax(output[5],axis=1).reshape((360,480)), False)


# cv2.imwrite('sdfs.jpg',gt[5])

# cv2.imwrite('haha.png',pred)

# figure = plt.figure(1)
# plt.imshow(pred)




# plt.imshow(pred)
# plt.figure(2)
# plt.imshow(gt[2])