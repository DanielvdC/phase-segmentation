#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:19:54 2021

@author: danielvdcorput
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
def find_clusters(frames):
    prev = None
    group = []
    for frame in frames:
        if not prev or frame-prev <= 1000:
            group.append(frame)
        else:
            yield group
            group = [frame]
        prev = frame
    if group:
        yield group

# define which video to assess
test_videos = os.path.join(get_parent_dir(1),'Data','videos','testset')
# load test video
videos = ['20180926I09.MP4','20180928I10.MP4','20181015I31.MP4',
          'ch1_video_001aa.mpg','ch1_video_001.mpg','ch1_video_05.mpg',
          '20180928I11_1.MP4']
vid_idx = 5
video = (os.path.join(test_videos, videos[vid_idx]))
print('[INFO] Currently processing video [{}] -- eager approach'.format(videos[vid_idx]))
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

p5 = 0
try:
    p5_start = convert_timelist_to_fps(test_label['phase5_start'].to_list(), fps) 
    p5_end = convert_timelist_to_fps(test_label['phase5_end'].to_list(), fps)
    p5_times = [range(start, stop+1) for start, stop in zip(p5_start, p5_end)]
    p5 = 1
except AttributeError:
    print('[INFO] Phase 5 not detected in ground truth labels -- successfully excluded from analysis.')

oob_times = [range(start, stop+1) for start, stop in zip(oob_start, oob_end)]
p1_times = [range(p1_start, p1_end+1)]
p2_times = [range(p2_start, p2_end+1)]
p3_times = [range(p3_start, p3_end+1)]
p4_times = [range(p4_start, p4_end+1)]

# create y_true label
y_true = []

if p5 == 0:
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
        else:
            y_true.append('nan')
elif p5 == 1:
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

# create complete dataframe
predictions['y_true'] = y_true
time = [frame/fps/60 for frame in predictions['frame']]
predictions['time'] = time

#%%
#### Standard operation:
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
MEDFILT = 53
median_filter = False
if median_filter:
    label = test['label'].to_list()
    label = onehotencoder(label)
    label = medfilt(label, MEDFILT)
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
if median_filter == True:
    p.set(title='Median filter [{}] predictions video [{}]'.format(MEDFILT,videos[vid_idx].split('.')[0]))
else:
    p.set(title='Raw prediction scores video [{}]'.format(videos[vid_idx].split('.')[0]))
p._legend.remove()
p.fig.set_size_inches(9.5, 3)
plt.show()

print('[INFO] Plotting instrument detections...')
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
p.fig.set_size_inches(15, 3)
print('[INFO] Completed.')

#%%
test = test.sort_values(by='frame')
print(test_label['phase3_start'])
print(test_label['phase3_end'])
print(test.loc[test[test['label'] == 'phase 3'].last_valid_index()])

#print(df.loc[df[df['Instrument'] == 'Clipper'].first_valid_index()])
#%%
##### APPLY APPROPRIATE RULES BASED ON PLOT #####
##### Rule no. 1: change predictions of phase 1 if it is after the first occurrence
##### of phase 2
test = test.sort_values(by='frame')
fo_2 = test[test['label'] == 'phase 2'].first_valid_index()
for index,row in test[test['label'] == 'phase 1'].iterrows():
    if index >= fo_2:
        preds = test['label'].loc[index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
        preds = [p for p in preds.index if p != 'phase 1']
        test.at[index,'label'] = preds[np.argmax(preds)]        

#%%
##### Rule no. 2: change predictions of phase 1 if they are outside of their 
##### first cluster to most common neighbours
frame_list = []
test = test.sort_values(by='frame')
for index,row in test[test['label'] == 'phase 1'].iterrows():
    frame_list.append(index)
frame_list = sorted(frame_list)

clusters = dict(enumerate(find_clusters(frame_list),1))

fo_1 = test[test['label'] == 'phase 1'].first_valid_index()
lo_1 = 5654
    
KERNEL_SIZE = 25
for index,row in test[test['label'] == 'phase 1'].iterrows():
    if index < fo_1 or index > lo_1:
        try:
            preds = test['label'].loc[index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
            preds = [p for p in preds.index if p != 'phase 1']
            test.at[index,'label'] = preds[np.argmax(preds)]
        except KeyError:
            test.at[index,'label'] = None
        except ValueError:
            test.at[index,'label'] = None
#%%
test = test.sort_values(by='frame')
fo_1 = test[test['label']=='phase 1'].first_valid_index()
lo_1 = test[test['label']=='phase 1'].last_valid_index()

for index,row in test.iterrows():
    if index > fo_1 and index < lo_1:
        test.at[index,'label'] = 'phase 1'

#%%
##### Rule no. 3: change predicitons to phase 1 if they are between the first
##### and last occurrence of phase 1
test = test.sort_values(by='frame')

frame_list = []
test = test.sort_values(by='frame')
for index,row in test[test['label'] == 'phase 1'].iterrows():
    frame_list.append(index)
frame_list = sorted(frame_list)

clusters = dict(enumerate(find_clusters(frame_list),1))
fo_1 = clusters[2][1]
lo_1 = clusters[2][-1]
#%%
test = test.sort_values(by='frame')

for index,row in test.iterrows():
    if index > lo_1 and index < fo_3:
        test.at[index,'label'] = 'phase 2'
        


#%%

#fo_1 = test[test['label'] == 'phase 1'].first_valid_index()
#fo_1 = test.loc[fo_1]['frame']
#lo_1 = test[test['label'] == 'phase 1'].last_valid_index()
#lo_1 = test.loc[lo_1]['frame']

for index,row in test.iterrows():
    if row['frame'] >= fo_1 and row['frame'] <= lo_1:
        test.at[index,'label'] = 'phase 1'
#%%
##### Rule no. 4: change predictions of phase 2 if it is before the last 
##### occurrence of phase 1
# first find appropriate end point of phase 1 based on clusters        
frame_list = []
test = test.sort_values(by='frame')
for index,row in test[test['label'] == 'phase 1'].iterrows():
    frame_list.append(index)
frame_list = sorted(frame_list)

clusters = dict(enumerate(find_clusters(frame_list),1))
#lo_1 = clusters[1][-1]


test = test.sort_values(by='frame')
for index,row in test[test['label'] == 'phase 2'].iterrows():
    if index <= lo_1:
        test.at[index,'label'] = 'phase 1'
#%%
##### Rule no. 5: change predictions to phase 2 if it is in between the last
##### occurence of phase 1 and the last occurrence of phase 2
test = test.sort_values(by = 'frame')
frame_list = []
for index,row in test[test['label'] == 'phase 1'].iterrows():
    frame_list.append(index)
frame_list = sorted(frame_list)


clusters = dict(enumerate(find_clusters(frame_list),1))
lo_1 = clusters[2][-1]
#%%
test = test.sort_values(by = 'frame')
frame_list = []
for index,row in test[test['label'] == 'phase 3'].iterrows():
    frame_list.append(index)
frame_list = sorted(frame_list)


clusters = dict(enumerate(find_clusters(frame_list),1))
lo_2 = clusters[2][-1]

#%%
fo_3 = test[test['label']=='phase 3'].first_valid_index()

for index,row in test.iterrows():
    #if row['label'] != 'oob':
        if index > lo_1 and index < fo_3:
            test.at[index,'label'] = 'phase 2'
#%%
test = test.sort_values(by = 'frame')
for index,row in test[test['label'] == 'phase 2'].iterrows():
    if index > lo_2:
        preds = test['label'].loc[index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
        preds = [p for p in preds.index if p != 'phase 2']
        test.at[index,'label'] = preds[np.argmax(preds)]        

#%%
##### Rule no. 6: change predictions of phase 2 if it is after the last 
##### occurrence of phase 3
test = test.sort_values(by = 'frame')
lo_3 = test[test['label'] == 'phase 3'].last_valid_index()
for index,row in test[test['label'] == 'phase 2'].iterrows():
    if index >= lo_3:
        test.at[index,'label'] = 'phase 3'
#%%
##### Rule no. 7: change predictions of phase 2 if it is after phase 3 based
##### on majority vote
KERNEL_SIZE = 25
test = test.sort_values(by = 'frame')
lo_3 = test[test['label'] == 'phase 3'].last_valid_index()
for index,row in test[test['label'] == 'phase 2'].iterrows():
    if index > lo_3:
        preds = test['label'].loc[index-KERNEL_SIZE:index+KERNEL_SIZE].value_counts()
        preds = [p for p in preds.index if p != 'phase 2']
        test.at[index,'label'] = preds[np.argmax(preds)]
#%%
##### Rule no. 8: use the first occurrence of the clipper as the starting point
##### for phase 3
test = test.sort_values(by = 'frame')
frame_list = []
df = df.sort_values(by='Frame ID')
for index,row in df[df['Instrument'] == 'Clipper'].iterrows():
    frame_list.append(index)
frame_list = sorted(frame_list)

clusters = dict(enumerate(find_clusters(frame_list),1))

print(clusters)
fo_3 = clusters[9][1]
lo_3 = clusters[9][-1]

fo_3 = df[df['Instrument'] == 'Clipper'].loc[clusters[9][1]['Frame ID']
print(fo_3)




#%%
test = test.sort_values(by = 'frame')

for index,row in test.iterrows():
    if row['frame'] > fo_3 and row['frame'] < lo_3:
        test.at[index,'label'] = 'phase 3'
        
#%%
fo_3 = test[test['label'] == 'phase 3'].first_valid_index()

for index,row in test.iterrows():
    if index > lo_1 and index < fo_3:
        test.at[index,'label'] = 'phase 2'


#%%
##### Rule no. 9: change predictions to phase 3 if it is between the last
##### occurence of the phase 2 cluster and the last occurrence of phase 3
test = test.sort_values(by = 'frame')
lo_3 = test[test['label'] == 'phase 3'].last_valid_index()

for index,row in test.iterrows():
    if index > lo_2 and index <= lo_3:
        test.at[index,'label'] = 'phase 3'
#%%
##### Rule no. 10: change predictions of phase 4 if it is before the last 
##### occurrence of phase 3
test = test.sort_values(by='frame')

lo_3 = test[test['label'] == 'phase 3'].last_valid_index()
for index,row in test[test['label'] == 'phase 4'].iterrows():
    if index < lo_3:
        test.at[index,'label'] = None
test = test.dropna()

#%%
##### Rule no. 11: change predictions of phase 4 if it is after the last occurence
##### of phase 5
test = test.sort_values(by = 'frame')
lo_5 = test[test['label'] == 'phase 5'].last_valid_index()
for index,row in test[test['label'] == 'phase 4'].iterrows():
    if index >= lo_5:
        test.at[index,'label'] = 'phase 5'
        
        

#%%
##### Rule no. 12: change predictions of phase 4 if they are after the last occurence
##### of phase 5
frame_list = []
for index,row in test[test['label'] == 'phase 4'].iterrows():
    frame_list.append(index)
frame_list = sorted(frame_list)

clusters = dict(enumerate(find_clusters(frame_list),1))
lo_4 = clusters[5][-1]

test = test.sort_values(by = 'frame')
for index,row in test[test['label'] == 'phase 4'].iterrows():
    if index >= lo_4:
        test.at[index,'label'] = 'phase 5'
        

        
#%%
##### Rule no. 13: change predictions to phase 4 if they are between the first
##### and last occurrence of phase 4
test = test.sort_values(by = 'frame')
fo_4 = test[test['label'] == 'phase 4'].first_valid_index()
lo_4 = test[test['label'] == 'phase 4'].last_valid_index()

for index,row in enumerate(test['label']):
    if index >= fo_4 and index <= lo_4:
        test.at[index,'label'] = 'phase 4'


#%%
##### Rule no. 14: change predictions of phase 5 if it is before the last 
##### occurrence of phase 4
test = test.sort_values(by = 'frame')
lo_4 = test[test['label'] == 'phase 4'].last_valid_index()
for index,row in test[test['label'] == 'phase 5'].iterrows():
    if index <= lo_4:
        test.at[index,'label'] = 'phase 4'
#%%
##### Rule no. 15: extra median filter
test = test.sort_values(by = 'frame')
test = test.dropna()
MEDFILT = 57
median_filter = True
if median_filter:
    label = test['label'].to_list()
    label = onehotencoder(label)
    label = medfilt(label, MEDFILT)
    label = reverse_encoder(label)
    test['label'] = label
#%%
##### Rule no. 16: change out of body predictions if they are between the first
##### and last occurrence of phase 1
test = test.sort_values(by = 'frame')

fo_1 = test[test['label'] == 'phase 1'].first_valid_index()
lo_1 = test[test['label'] == 'phase 1'].last_valid_index()
lo_2 = test[test['label'] == 'phase 2'].last_valid_index()

for index,row in test[test['label'] == 'oob'].iterrows():
    if row['frame'] > fo_1 and row['frame'] < lo_1:
        test.at[index,'label'] = 'phase 1'
    elif row['frame'] > lo_1 and row['frame'] < lo_2:
        test.at[index,'label'] = 'phase 2'
        
        

#%%
# Plot predictions
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
if median_filter == True:
    p.set(title='Median filter [{}] predictions video [{}]'.format(MEDFILT,videos[vid_idx].split('.')[0]))
else:
    p.set(title='Raw prediction scores video [{}]'.format(videos[vid_idx].split('.')[0]))

p._legend.remove()
p.fig.set_size_inches(9.5, 3)
plt.show()  
print('[INFO] Completed.')
        
#%%
# Plot phase transitions for ground truth
phases = ['phase 1','phase 2','phase 3','phase 4','phase 5']
fo = [test[test['y_true'] == ph].first_valid_index() for ph in phases]
fo = [f for f in fo if f]
fo = [test.loc[f]['time'] for f in fo]
fo.append(test.loc[test[test['y_true'] == phases[len(fo)-1]].last_valid_index()]['time'])

test = test[test['y_true'] != 'nan']
test = test.sort_values(by = 'frame')

q = sns.catplot(x = 'time',
                y = 'y_true',
                hue = 'y_true',
                data = test,
                jitter = False)

q.set(xlabel=None)
q.set(ylabel=None)
q._legend.remove()
q.fig.set_size_inches(9.5, 3)
for ax in q.axes.flat:
    for i,o in enumerate(fo):
        ax.axvline(o, ls=':', c='black')
        ax.text(o, -1.2, '{0:02.0f}:{1:02.0f}'.format(*divmod(o * 60, 60)),
                rotation=60, verticalalignment='center')
        ax.text(x=max(test['time'])/2, y=5.5,
                s='Ground truth phases video [{}] -- eager approach'.format(videos[vid_idx].split('.')[0]),
                fontsize=14, alpha=0.4,
                ha='center', va='bottom')

# Plot phase transitions for predictions
##### Find phase transitions
test = test.sort_values(by = 'frame')

phases = ['phase 1','phase 2','phase 3','phase 4','phase 5']
fo = [test[test['label'] == ph].first_valid_index() for ph in phases]
fo = [f for f in fo if f]
fo = [test.loc[f]['time'] for f in fo]
fo.append(test.loc[test[test['label'] == phases[len(fo)-1]].last_valid_index()]['time'])

test = test.sort_values(by = 'frame')

p = sns.catplot(x = 'time',
                y = 'label',
                hue = 'label',
                data = test,
                jitter = False)
p._legend.remove()
p.set(xlabel = None)
p.set(ylabel = None)
p.fig.set_size_inches(9.5, 3)
for ax in p.axes.flat:
    for i,o in enumerate(fo):
        ax.axvline(o, ls=':', c='black')
        ax.text(o, -1.1, '{0:02.0f}:{1:02.0f}'.format(*divmod(o * 60, 60)),
                rotation=50, verticalalignment='center')
        ax.text(x=max(test['time'])/2, y=5.5,
                s='Predicted phases video [{}] -- eager approach'.format(videos[vid_idx].split('.')[0]),
                fontsize=14, alpha=0.4,
                ha='center', va='bottom')
        
#%%





#%%


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
p.fig.set_size_inches(15, 3)
print('[INFO] Completed.')
    
