#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:01:30 2021

@author: danielvandencorput
"""
from tensorflow import keras
import pandas as pd
import numpy as np
import cv2
import os
import time

###############################################################################
############################# VIDEO PREDICTIONS ###############################
###############################################################################

# This script is used to generate the predictions for the videos. Make sure to
# add your videos to the ./Data/videos/testset/ directory and to add your ground
# truth labels to the ./Data/data_annotation/ directory. The script can also be
# alterated so that the ground truth labels are not necessary.
# Generates a .csv file that is used for the post processing in the other files.

###############################################################################

def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

# locate model
model_path = os.path.join(get_parent_dir(1),'Output', 'models')
# load model
models = ['efficientnetb3_0.01.model','efficientnetb3_0.05.model','efficientnetb3_0.25.model']
model_idx = 0
model = keras.models.load_model(os.path.join(model_path, models[model_idx]))

# locate test path
test_path = os.path.join(get_parent_dir(1),'Data','videos','testset')
# load test video
videos = [#YOUR VIDEOS HERE]
vid_idx = 0
video = (os.path.join(test_path, videos[vid_idx]))
# locate gt labels
gt_labels = os.path.join(get_parent_dir(1),'Data','data_annotation','gt_labels.csv')
# load gt labels
gt_labels = pd.read_csv(gt_labels, sep=';')
gt_labels = gt_labels.iloc[-3:]
classes = ['oob','phase 1','phase 2','phase 3','phase 4','phase 5']
# output dataframe
output = pd.DataFrame(columns=['frame','y_pred','y_true'])
# output path
output_file = os.path.join(get_parent_dir(1),'Output','{}_{}.csv'
                           .format(models[model_idx].split('model')[0][:-1],
                                   videos[vid_idx].split('.')[0]))
# instantiate video
cap = cv2.VideoCapture(video)
fps = cap.get(5)

# read video
while cap.isOpened():
    # set start time
    start = time.time()
    # read frame
    ret, frame = cap.read()
    # exceptions
    if ret == False:
        break
    if frame is None:
        continue
    # track currentframe
    currentframe = int(cap.get())
    # resize and expand to include batch size
    frame = cv2.resize(frame, (224,224))
    frame = np.expand_dims(frame, axis=0)
    # retrieve frame number
    name = str(currentframe)
    # make prediction over frame
    prediction = model.predict(frame)
    # convert highest score to class label
    predidx = np.argmax(prediction)
    pred = classes[predidx]
    # create list of predictions
    out = []
    [out.append({'frame':name,'y_pred':pred,'y_true':0})]
    output = output.append(out)
    if currentframe % 25 == 0:
        print(currentframe)
    
output.to_csv(output_file,header=True,index=False)
