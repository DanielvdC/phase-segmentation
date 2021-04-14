#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:56:51 2021

@author: danielvandencorput
"""
import tensorflow as tf
from imutils import paths
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random
import os
plt.use('Agg')

def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

### Set up dataset
SKIP_DATASETS = False
# full dataset
ORIG_INPUT_DATASET = os.path.join(get_parent_dir(1),'Data','images','full dataset')
# train / validation / test paths
BASE_PATH = os.path.join(get_parent_dir(1),'Data','images')
TRAIN_PATH = os.path.sep.join([BASE_PATH, 'train'])
VAL_PATH = os.path.sep.join([BASE_PATH, 'valid'])
TEST_PATH = os.path.sep.join([BASE_PATH, 'test'])

# define splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.1
# define classes
CLASSES = ['oob','phase_1','phase_2','phase_3','phase_4','phase_5']

# initial learning rate, batch size and number of epochs
INIT_LR = 1e-4
BS = 32
NUM_EPOCHS = 500
FULL_DATA_PERCENTAGE = 0.01
# model path
MODEL_PATH = os.path.join(get_parent_dir(1),'Output','efficientnetb3_{}.model'.format(FULL_DATA_PERCENTAGE))

# grab paths of all input images, shuffle and take sample
imagePaths = list(paths.list_images(ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)
imagePaths = random.sample(imagePaths, int(len(imagePaths)*FULL_DATA_PERCENTAGE))

# set up training and testing split
i = int(len(imagePaths) * TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# set up validation split from training split
i = int(len(trainPaths) * VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# define datasets
datasets = [
	("training", trainPaths, TRAIN_PATH),
	("validation", valPaths, VAL_PATH),
	("testing", testPaths, TEST_PATH)
]

if SKIP_DATASETS == True:
    print('[INFO] Skipping dataset creation')
    pass
else:
    print('[INFO] Creating datasets...')
    # loop over datasets
    for (dType, imagePaths, baseOutput) in datasets:
        print('[INFO] building {} split'.format(dType))
        # if output directory does not exist, create it
        if not os.path.exists(baseOutput):
            print("[INFO] 'creating {}' directory".format(baseOutput))
            os.makedirs(baseOutput)
    	# loop over input image paths
        for inputPath in imagePaths:
            #extract the filename of the input image along with its label
            filename = inputPath.split(os.path.sep)[-1]
            label = inputPath.split(os.path.sep)[-2]
            # build the path to the label directory
            labelPath = os.path.sep.join([baseOutput, label])
            # if the label output directory does not exist, create it
            if not os.path.exists(labelPath):
                print("[INFO] 'creating {}' directory".format(labelPath))
                os.makedirs(labelPath)
            # construct the path to the destination image and then copy
            # the image itself
            p = os.path.sep.join([labelPath, filename])
            shutil.copy2(inputPath, p)

### Load data generators, load model and prepare for fine-tuning
# get total number of image paths
totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))
totalTest = len(list(paths.list_images(TEST_PATH)))

# set up data generators without any augmentation since this is already done
# during data collection
trainAug = ImageDataGenerator()
valAug = ImageDataGenerator()

# define ImageNet mean subtraction (RGB) and set for each generator
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	TRAIN_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	VAL_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	TEST_PATH,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# load Efficientnetb3
print("[INFO] preparing model...")
baseModel = EfficientNetB3(weigths = 'imagenet',
                           include_top = False,
                           input_tensor = Input(shape=(224, 224, 3)))
# construct head of model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(256, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(6, activation='softmax')(headModel)
# add them together
model = Model(inputs = baseModel.input, outputs = headModel)
# freeze original layers
for layer in baseModel.layers:
    layer.trainable = False

### Compile and train the model
# compile the model
opt = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

ACCURACY_THRESHOLD = 0.99
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > ACCURACY_THRESHOLD):
            print("\n[INFO] Reached %2.2f%% accuracy, so stopping training..." %(ACCURACY_THRESHOLD*100))
            self.model.stop_training = True
callbacks = myCallback()

# train the model
print("[INFO] training model...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	epochs=NUM_EPOCHS,
    callbacks = [callbacks])

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(os.path.join(get_parent_dir(1),'Output','loss_accuracy_efficientnetb3_{}split.png'.format(FULL_DATA_PERCENTAGE)))

### Test model on test data
print("[INFO] evaluating network...")
# reset test generator
testGen.reset()
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // BS) + 1)
# find index of label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show classification report
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))
# serialize the model to disk
print("[INFO] saving model...")
model.save(MODEL_PATH, save_format="h5")

