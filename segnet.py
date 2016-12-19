from __future__ import print_function
import keras
from keras.callbacks import TensorBoard
from keras.layers.noise import GaussianNoise
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf
import cv2
import numpy as np
import os
import glob


# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# ]

PTD = 'path_to_dataset'       # path to data folder
PTT = PTD + '/train'
PTL = PTD + '/label'
NI = 3470       # number of images in the training set
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

# void        0   0   0   0
# sky         70  130 180 1 -
# Building    70  70  70  2 -
# Road        128 64  128 3 -
# Sidewalk    244 35  232 4 -
# Fence       64  64  128 5 + 190,153,153
# Vegetation  107 142 35  6 -
# Pole        153 153 153 7 -
# Car         0   0   142 8 -
# Trafficsign 220 220 0   9 -
# Pedestrian  220 20  60  10-
# Bicycle     119 11  32  11-
# Motorcycle  0   0   230 12-
# Parking-slot250 170 160 13+ 81, 0, 81 to ground
# Road-work   128 64  64  14+ 190,153,153 to fence
# Traffic lit 250 170 30  15-
# Terrain     152 251 152 16-
# Rider       255 0   0   17-
# Truck       0   0   70  18-
# Bus         0   60  100 19-
# Train       0   80  100 20-
# Wall        102 102 156 21-
# Lanemarking 102 102 156 22+ 128 ,64, 128 to road 
# --- bridge, tunnel van, 

URBAN_CLASS = [void, dynamic, ground, road, sidewalk, building, wall, fence, bridge, tunnel, pole, tlight, tsign, vegetation,
			   terrain, sky, pedestrian, rider, car, truck, bus, van, trailer, train, motorcycle, bicycle]
URBAN_CLASS = np.array(URBAN_CLASS)

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


# data generator for large data sets
def generate_arrays_from_file(path_to_train, path_to_label):
	ptt = path_to_train
	ptl = path_to_label
	train_imgs = glob.glob(PTT + '/train_'+'*.png')
	label_imgs = glob.glob(PTL + '/train_'+'*.png')
	while True:
		for i in xrange(NI):
			train_data = np.array([np.rollaxis(normalized(cv2.imread(train_imgs[i])),2)])
			label_img = cv2.imread(label_imgs[i])
			temp_label = np.zeros((IW * IL, len(URBAN_CLASS)))
			for x in xrange(IW):
				for y in xrange(IL):
					temp_label[x*IL+y][label_img[x][y][0]] = 1
			train_label = np.array([temp_label])
			yield (train_data, train_label)

# one-time data array for small data sets
# # loading train data
# print ("Start loading training data")
# train_imgs = glob.glob(PTT + '/*.png')
# train_data = []
# i = 0
# while i < NI:
#     train_data += [np.rollaxis(normalized(cv2.imread(train_imgs[i])),2)]
#     # print (i/(NI*1.0),"percent image loaded\r",)
#     i += 1
# train_data = np.array(train_data)
# print ("train data loaded, shape:" ,train_data.shape)

# print ("Start loading training label")
# label_imgs = glob.glob(PTL + '/*.png')
# train_label = []
# i = 0
# while i < NI:
#     label_img = cv2.imread(label_imgs[i])
#     train_label += [[]]
#     temp_label = np.zeros((IW * IL, len(URBAN_CLASS)))
#     for x in xrange(IW):
#         for y in xrange(IL):
#             temp_label[x*IL+y][label_img[x][y][0]] = 1
#             train_label[i] = temp_label
#     i += 1
# train_label = np.array(train_label)
# print ("train label loaded, shape:" ,train_label.shape)


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
model.add(ZeroPadding2D(padding=(1,1), dim_ordering='th'))
model.add(Convolution2D(128,3,3, border_mode='valid', dim_ordering='th'))
model.add(BatchNormalization())
model.add(Activation('relu'))
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

print (model.summary())

model.compile(loss="categorical_crossentropy", optimizer='adadelta',metrics=["accuracy"])

nb_epoch = 400
# batch_size = 5
sp_epoch = 1500

try:

	tb = TensorBoard(log_dir='path_to_tensorboard_log',histogram_freq=1, write_graph=True, write_images=False)
	model.fit_generator(generate_arrays_from_file(PTT, PTL), nb_epoch=nb_epoch, samples_per_epoch=sp_epoch,
						nb_worker=10 ,show_accuracy=True, verbose=1, callbacks=[tb])
except KeyboardInterrupt:
	pass


model.save_weights('ep100.h5')
del model  # deletes the existing model

