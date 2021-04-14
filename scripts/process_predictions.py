#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 09:34:15 2021

@author: danielvandencorput
"""
from scipy.signal import medfilt

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import os

def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path
def onehotencoder(pred_list):
    els = ['oob','phase 1','phase 2','phase 3','phase 4','phase 5']
    reps = [0,1,2,3,4,5]
    for i in range(len(pred_list)):
        idx = els.index(pred_list[i])
        pred_list[i] = reps[idx]
    return pred_list
def reverse_encoder(pred_list):
    out_list = []
    els = [0,1,2,3,4,5]
    reps = ['oob','phase 1','phase 2','phase 3','phase 4','phase 5']
    for p in range(len(pred_list)):
        idx = els.index(pred_list[p])
        out_list.append(reps[idx])
    return out_list

# define which video to assess
test_videos = os.path.join(get_parent_dir(1),'Data','videos','testset')
# load test video
videos = ['20180926I09.MP4','20180928I10.MP4','20181015I31.MP4',
          'ch1_video_001aa.mpg','ch1_video_001.mpg','ch1_video_05.mpg',
          '20180928I11_1.MP4']
vid_idx = 0
video = (os.path.join(test_videos, videos[vid_idx]))

# instantiate video to get the fps
cap = cv2.VideoCapture(video)
fps = cap.get(5)


# select model
models = ['resnet50v2_0.01.model','alexnet_0.01.model','efficientnetb3_0.01.model',
          'efficientnetb3_0.05.model','efficientnetb3_0.25.model']
model_idx = 4
# locate predictions
predictions = os.path.join(get_parent_dir(1),'Output','predictions',
                           '{}_{}.csv'.format(
                               models[model_idx].split('model')[0][:-1],
                               videos[vid_idx].split('.')[0]))
### Y PRED
# read predictions
predictions = pd.read_csv(predictions)
# create list of predictions
y_pred = predictions['y_pred'].to_list()
# one hot encode
y_pred = onehotencoder(y_pred)
# apply median filter
y_pred = medfilt(y_pred, 13)
# reverse encode
y_pred = reverse_encoder(y_pred)
# add to dataframe
predictions['y_pred'] = y_pred

### Y TRUE
# locate gt labels
gt_labels = os.path.join(get_parent_dir(1),'Data','data_annotation','gt_labels.csv')
# load gt labels
gt_labels = pd.read_csv(gt_labels, sep=';')

# set up ground truth for correct video
test_label = gt_labels[gt_labels['Filename']==videos[vid_idx].split('.')[0]]

# convert timestamps to correct fps
def convert_timelist_to_fps(times,fps):
    for time in times:
        time_fps_list = [int(int((int(t.split(':')[0])*60) + int(t.split(':')[1])) * fps)
                         for t in time.split(';')]
    return time_fps_list

def convert_to_fps(time, fps):
    time = int((int(time.split(':')[0])*60 + int(time.split(':')[1])) * fps)
    return time

# get time ranges
oob_start = convert_timelist_to_fps(test_label['phase6_start'].to_list(), fps)
oob_end = convert_timelist_to_fps(test_label['phase6_end'].to_list(), fps)
p1_start = convert_to_fps(test_label['phase1_start'].item(), fps) 
p1_end = convert_to_fps(test_label['phase1_end'].item(), fps)
p2_start = convert_to_fps(test_label['phase2_start'].item(), fps) 
p2_end = convert_to_fps(test_label['phase2_end'].item(), fps)
p3_start = convert_to_fps(test_label['phase3_start'].item(), fps) 
p3_end = convert_to_fps(test_label['phase3_end'].item(), fps)
p4_start = convert_to_fps(test_label['phase4_start'].item(), fps) 
p4_end = convert_to_fps(test_label['phase4_end'].item(), fps)
p5_start = convert_timelist_to_fps(test_label['phase5_start'].to_list(), fps) 
p5_end = convert_timelist_to_fps(test_label['phase5_end'].to_list(), fps)

oob_times = [range(start, stop+1) for start, stop in zip(oob_start, oob_end)]
p1_times = [range(p1_start, p1_end+1)]
p2_times = [range(p2_start, p2_end+1)]
p3_times = [range(p3_start, p3_end+1)]
p4_times = [range(p4_start, p4_end+1)]
p5_times = [range(start, stop+1) for start, stop in zip(p5_start, p5_end)]

# create y_true label
y_true = []

for i in range(0, predictions['frame'].max()):
    if any([i in my_range for my_range in oob_times]):
        y_true.append('oob')
    elif any([i in my_range for my_range in p1_times]):    
        y_true.append('phase 1')
    elif any([i in my_range for my_range in p2_times]):    
        y_true.append('phase 2')       
    elif any([i in my_range for my_range in p3_times]):    
        y_true.append('phase 3')  
    elif any([i in my_range for my_range in p4_times]):    
        y_true.append('phase 4')
    elif any([i in my_range for my_range in p5_times]):    
        y_true.append('phase 5')
    else:
        y_true.append('nan')


predictions['y_true'] = y_true
time = [frame/fps/60 for frame in predictions['frame']]
predictions['time'] = time
#%%
####
# change phase prediction by most common neighbour if they are before the
# first occurence of the previous phase.
KERNEL_SIZE = 100
CONF_SCORE = 0.9
phases = ['phase 2','phase 3','phase 4','phase 5']

for phase in range(len(phases)):
    if phase == 0:
        fo = predictions[predictions['y_pred'] == 'phase 1'].first_valid_index()
    for index,row in predictions[predictions['y_pred'] == phases[phase]].iterrows():
        if row['frame'] <= fo:
            preds = predictions['y_pred'][index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
            preds = [p for p in preds.index if p != phases[phase]]
            predictions.at[index,'y_pred'] = preds[np.argmax(preds)]
    fo = predictions[predictions['y_pred'] == phases[phase]].first_valid_index()
label = [predictions.iloc[i]['y_pred'] if predictions.iloc[i]['conf_score'] >=
         CONF_SCORE else 'nan' for i in range(len(predictions))]
predictions['label'] = label

#%%
### Plot ground truth labels and the altered predictions
test = predictions.copy()
# remove nan's and sort values 
test = test[test.y_true != 'nan']
test = test.sort_values(by='y_true')
# plot actual labels
q = sns.catplot(x = 'time',
                y = 'y_true',
                hue = 'y_true',
                data = test,
                jitter = False)
q.set(ylabel=None)
q.set(xlabel=None)
q.set(title='Actual labels video [{}]'.format(videos[vid_idx].split('.')[0]))
q._legend.remove()
q.fig.set_size_inches(9.5, 3)

test = predictions.copy()
test = test[test['label'] != 'nan']
# apply median filter
median_filter = True
if median_filter:
    label = test['label'].to_list()
    label = onehotencoder(label)
    label = medfilt(label, 157)
    label = reverse_encoder(label)
    test['label'] = label

print('[INFO] Plotting predictions...')
# sort values
test = test.sort_values(by = 'label')

# plot predictions
p = sns.catplot(x = 'time',
                y = 'label',
                hue = 'label',
                data = test,
                jitter = False)
p.set(ylabel=None)
p.set(xlabel=None)
p.set(title='Median filter [157] predictions video [{}]'.format(videos[vid_idx].split('.')[0]))
p._legend.remove()
p.fig.set_size_inches(9.5, 3)
plt.show()  

print('[INFO] Completed.')

#%%
### loop through phase 4 and phase 5 and change them with majority vote if they 
### are before the last occurence of the previous phase
phases = ['phase 4','phase 5']

KERNEL_SIZE = 20
        
lo_3 = predictions[predictions['label'] == 'phase 3'].last_valid_index()
for index,row in predictions[predictions['label'] == 'phase 4'].iterrows():
    if row['frame'] <= lo_3:
        preds = predictions['label'][index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
        preds = [p for p in preds.index if p != 'phase 4']
        predictions.at[index,'label'] = preds[np.argmax(preds)]
lo_4 = predictions[predictions['label'] == 'phase 4'].last_valid_index()
for index,row in predictions[predictions['label'] == 'phase 5'].iterrows():
    if row['frame'] <= lo_4:
        preds = predictions['label'][index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
        preds = [p for p in preds.index if p != 'phase 5']
        predictions.at[index,'label'] = preds[np.argmax(preds)]
        
test = predictions.copy()
test = test[test['label'] != 'nan']

median_filter = True
if median_filter:
    label = test['label'].to_list()
    label = onehotencoder(label)
    label = medfilt(label, 97)
    label = reverse_encoder(label)
    test['label'] = label
  
print('[INFO] Plotting predictions...')
# sort values
test = test.sort_values(by = 'label')
# plot predictions
p = sns.catplot(x = 'time',
                y = 'label',
                hue = 'label',
                data = test,
                jitter = False)
p.set(ylabel=None)
p.set(title='Rule based enhancement 2.2.1 [{}]'.format(videos[vid_idx].split('.')[0]))
p._legend.remove()
p.fig.set_size_inches(12, 3)

#%%
### 2.1.1
# Change phase prediction if it is before first occurence of previous phase
# Next prediction should always be the same phase or phase+1

KERNEL_SIZE = 100
CONF_SCORE = 0.9
transitions = ['phase 2','phase 3','phase 4','phase 5']
phases = ['phase 1','phase 2','phase 3','phase 4','phase 5']

for trans in range(len(transitions)):
    if trans == 0:
        fo = predictions[predictions['y_pred'] == 'phase 1'].first_valid_index()
    for index,row in predictions[predictions['y_pred'] == transitions[trans]].iterrows():
        try:
            if row['frame'] <= fo:
                prev_pred = predictions['y_pred'][index-1]
            if prev_pred != 'oob' and row['y_pred'] != prev_pred and row['y_pred'] != phases[phases.index(prev_pred)+1]:
                preds = predictions['y_pred'][index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
                preds = [p for p in preds.index if p == prev_pred or p == phases[phases.index(prev_pred)+1]]
                predictions.at[index,'y_pred'] = preds[np.argmax(preds)]
            fo = predictions[predictions['y_pred'] == transitions[trans]].first_valid_index()
        except IndexError:
                    preds = predictions['y_pred'][index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
                    preds = [p for p in preds.index if p == prev_pred or p == phases[
                        phases.index(prev_pred)]]
                    predictions.at[index,'y_pred'] = preds[np.argmax(preds)] 
                    fo = predictions[predictions['y_pred'] == transitions[trans]].first_valid_index()

            
#%%
# plot predictions
print('[INFO] Plotting predictions...')
# sort values
predictions = predictions.sort_values(by = 'y_pred')

median_filter = True
if median_filter:
    y_pred = predictions['y_pred'].to_list()
    y_pred = onehotencoder(y_pred)
    y_pred = medfilt(y_pred, 57)
    y_pred = reverse_encoder(y_pred)
    predictions['y_pred'] = y_pred
    
# plot predictions
p = sns.catplot(x = 'time', 
                y = 'y_pred',
                hue = 'y_pred',
                data = predictions,
                jitter = False)
p.set(ylabel=None)
p.set(xlabel=None)
p.set(title='Rule based enhancement 2.1.3 video [{}]'.format(videos[vid_idx].split('.')[0]))
p._legend.remove()
p.fig.set_size_inches(10.5, 3)

#%%
### change predictions if they are before the last occurence of the previous phase
### change by using rational alteration
phases = ['phase 1','phase 2','phase 3','phase 4','phase 5']

KERNEL_SIZE = 20

median_filter = True
if median_filter:
    y_pred = predictions['y_pred'].to_list()
    y_pred = onehotencoder(y_pred)
    y_pred = medfilt(y_pred,97)
    y_pred = reverse_encoder(y_pred)
    predictions['y_pred'] = y_pred

lo_3 = predictions[predictions['y_pred'] == 'phase 3'].last_valid_index()
for index,row in predictions[predictions['y_pred'] == 'phase 4'].iterrows():
    if row['frame'] <= lo_3:
        prev_pred = predictions['y_pred'][index-1]
        try:
            if prev_pred != 'oob' and row['y_pred'] != prev_pred and row['y_pred'] != phases[phases.index(prev_pred)+1]:
                preds = predictions['y_pred'][index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
                preds = [p for p in preds.index if p == prev_pred or p == phases[phases.index(prev_pred)+1]]
                #predictions.at[index,'y_pred'] = preds[np.argmax(preds)]    
        except IndexError:
            preds = predictions['y_pred'][index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
            preds = [p for p in preds.index if p == prev_pred or p == 'phase 5']
           # predictions.at[index,'y_pred'] = preds[np.argmax(preds)]  
#%%

lo_2 = predictions[predictions['label'] == 'phase 2'].last_valid_index()
#for index,row in predictions[predictions['label'] == 'phase 3'].iterrows():
#    if row['frame'] <= lo_2:
#        predictions.at[index,'label'] = 'phase 2'
        
copy = predictions.copy()
copy = copy[copy['label'] != 'nan']
copy = copy.sort_values(by='label')
# plot predictions
p = sns.catplot(x = 'time', 
                y = 'label',
                hue = 'label',
                data = copy,
                jitter = False)
p.set(ylabel=None)
p.set(xlabel=None)
p.set(title='Phase 3 alterations video [{}]'.format(videos[vid_idx].split('.')[0]))
p._legend.remove()
p.fig.set_size_inches(10.5, 3)
#%%
################################### 
# Alterations based on instrument detections
output_folder = os.path.join(get_parent_dir(1),'Output')
data_folder = os.path.join(get_parent_dir(1),'Data','videos','testset')
csv_file = 'instrument_detections_{}.csv'.format(videos[vid_idx].split('.')[0])
csv_file = os.path.join(output_folder, csv_file)
df = pd.read_csv(csv_file)

INSTRUMENTS_OUT = True
CAPITAL_LETTERS = True
THRESHOLD = 0.8


cap = cv2.VideoCapture(os.path.join(data_folder,videos[vid_idx]))
fps = cap.get(5)

time = [frame/fps/60 for frame in df['Frame ID']]
df['time'] = time

instruments = df['Instrument'].to_list()
#print(np.unique(instruments))

INSTRUMENTS_OUT = ['bipolar','gauze',
                   'irrigator','nan',
                   'needle_holder','stapler']

if INSTRUMENTS_OUT:
    for instrument in INSTRUMENTS_OUT:
        df.loc[df['Instrument'] == instrument] = None
if CAPITAL_LETTERS:
    df["Instrument"].replace({'scissors': "Scissors", 'dissection_device': "Dissection_Device", 
                                     'hook': "Hook", 'trocar': "Trocar",
                                     'irrigator': "Irrigator", 'clipper': "Clipper",
                                     'grasper': "Grasper", 'Out_of_body': 'Out of body',
                                     'specimen_bag': "Specimen bag", 'bipolar': 'Bipolar'}, inplace=True)
df = df.loc[df['confidence'] > THRESHOLD]


p = sns.catplot(x = 'time',
                y = 'Instrument',
                hue = 'Instrument',
                data = df,
                jitter = False,
                order=(['Trocar','Grasper','Hook','Dissection_Device',
                        'Clipper','Scissors','Specimen bag']))
p.set(ylabel=None)
p.set(xlabel=None)
p.set(title='Instrument detections video [{}]'.format(videos[vid_idx].split('.')[0]))
p._legend.remove()
p.fig.set_size_inches(21, 3.5)

#%%
# Remove instrument detections before the first detection OTHER THAN out of body
for index,row in enumerate(predictions['y_pred']):
    if row != 'oob':
        start_p1 = index
        break


start_p1 = predictions.iloc[start_p1]['frame']
df.loc[df['Frame ID'] < start_p1, 'Instrument'] = None

p = sns.catplot(x = 'time',
                y = 'Instrument',
                hue = 'Instrument',
                data = df,
                jitter = False,
                order=(['Trocar','Grasper','Hook','Dissection_Device',
                        'Clipper','Scissors','Specimen bag']))
p.set(ylabel=None)
p.set(xlabel=None)
p.set(xlim=(0,24))
p.set(title='3.1.1 Instrument detections video [{}]'.format(videos[vid_idx].split('.')[0]))
p._legend.remove()
p.fig.set_size_inches(21, 3.5)
#%%
# 3.1.2 Loop over phase 2 predictions and change if they are before the first occurrence
# of the grasper
KERNEL_SIZE = 25
fo_grasper = df[df['Instrument'] == 'Grasper'].first_valid_index()
fo_grasper = df.loc[fo_grasper]['Frame ID']

for index,row in predictions[predictions['y_pred'] == 'phase 2'].iterrows():
    if row['frame'] <= fo_grasper:
        preds = predictions['y_pred'][index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
        preds = [p for p in preds.index if p != 'phase 2']
        predictions.at[index,'y_pred'] = preds[np.argmax(preds)]
        
        
predictions = predictions.sort_values(by = 'y_pred')

p = sns.catplot(x = 'time',
                y = 'y_pred',
                hue = 'y_pred',
                data = predictions,
                jitter = False)
p.set(ylabel=None)
p.set(xlabel=None)
p.set(title='3.1.2 Phase 2 alteration video [{}]'.format(videos[vid_idx].split('.')[0]))
p._legend.remove()
p.fig.set_size_inches(11, 3)

#%%
# 3.1.3 Loop over phase 3 predictions and change if they are before the first occurrence
# of the clipper
plot = predictions.copy()
fo_clipper = df[df['Instrument'] == 'Clipper'].first_valid_index()
fo_clipper = df.loc[fo_clipper]['Frame ID']

for index,row in plot[plot['y_pred'] == 'phase 3'].iterrows():
    if row['frame'] <= fo_clipper:
        preds = plot['y_pred'][index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
        preds = [p for p in preds.index if p != 'phase 3']
        plot.at[index,'y_pred'] = preds[np.argmax(preds)]
        
median_filter = True
if median_filter:
    y_pred = plot['y_pred'].to_list()
    y_pred = onehotencoder(y_pred)
    y_pred = medfilt(y_pred,57)
    y_pred = reverse_encoder(y_pred)
    plot['y_pred'] = y_pred

plot = plot.sort_values(by = 'y_pred')        

p = sns.catplot(x = 'time',
                y = 'y_pred',
                hue = 'y_pred',
                data = plot,
                jitter = False)
p.set(ylabel=None)
p.set(xlabel=None)
p.set(title='3.1.3 Phase 3 alteration video [{}]'.format(videos[vid_idx].split('.')[0]))
p._legend.remove()
p.fig.set_size_inches(11,3)

        

#%%
copy = predictions.copy()

median_filter = True
if median_filter:
    y_pred = copy['y_pred'].to_list()
    y_pred = onehotencoder(y_pred)
    y_pred = medfilt(y_pred, 57)
    y_pred = reverse_encoder(y_pred)
    copy['y_pred'] = y_pred

copy = copy.sort_values(by='y_pred')
# plot predictions
p = sns.catplot(x = 'time', 
                y = 'y_pred',
                hue = 'y_pred',
                data = copy,
                jitter = False)
p.set(ylabel=None)
p.set(xlabel=None)
p.set(title='Pre processing inspection video [{}]'.format(videos[vid_idx].split('.')[0]))
p._legend.remove()
p.fig.set_size_inches(10.5, 3)

#%%
copy = predictions.copy()

median_filter = True
if median_filter:
    y_pred = copy['y_pred'].to_list()
    y_pred = onehotencoder(y_pred)
    y_pred = medfilt(y_pred, 57)
    y_pred = reverse_encoder(y_pred)
    copy['y_pred'] = y_pred

phases = ['phase 1','phase 2','phase 3','phase 4','phase 5']
current_phase = 0

for index,row in enumerate(copy['y_pred']):
    try:        
        if row == 'oob':
            continue
        if row == phases[current_phase]:
            continue
        if row == phases[current_phase + 1]:
            current_phase += 1
        if row == 'phase 5':
            if current_phase == 4:
                continue
            if current_phase == 3:
                current_phase += 1
            else:
                copy.at[index,'y_pred'] = phases[current_phase]
        else:
            copy.at[index,'y_pred'] = phases[current_phase]
    except IndexError:
        if row != 'oob':
            copy.at[index,'y_pred'] = phases[current_phase]

    



#%%
copy = predictions.copy()

median_filter = True
if median_filter:
    y_pred = copy['y_pred'].to_list()
    y_pred = onehotencoder(y_pred)
    y_pred = medfilt(y_pred, 157)
    y_pred = reverse_encoder(y_pred)
    copy['y_pred'] = y_pred
    
copy = copy.sort_values(by='y_pred')
# plot predictions
p = sns.catplot(x = 'time', 
                y = 'y_pred',
                hue = 'y_pred',
                data = copy,
                jitter = False)
p.set(ylabel=None)
p.set(xlabel=None)
p.set(title='Rational phase continuation video [{}]'.format(videos[vid_idx].split('.')[0]))
p._legend.remove()
p.fig.set_size_inches(10.5, 3)



    






