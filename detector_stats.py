# -*- coding: utf-8 -*-

# Author: Gopikrishnan Sasi Kumar

# This script tests the cascaded detector saved in 
# trained_cascaded_detector.pkl

import pickle
import library_adaboost_hw10 as lib
import matplotlib.pyplot as plt
dpi_value = 600

with open('trained_cascaded_detector.pkl', 'rb') as input_file:
    Fi_track = pickle.load(input_file)
    Di_track = pickle.load(input_file)
    cascade_det = pickle.load(input_file)
    
x_ax = list(range(1, len(Fi_track)))
fig = plt.figure()
plt.plot(x_ax, Fi_track[1:], color='red', marker='x', markersize = 5, \
         label = 'False Positive Rate')
plt.plot(x_ax, Di_track[1:], color='blue', marker='x', markersize = 5, \
         label = 'Detection Rate')
plt.xlabel('Cascade Stage')
plt.ylabel('Rate')
plt.xticks(x_ax)
plt.legend()
plt.savefig('Cascade_detector_train_rates.png', dpi = dpi_value)
plt.close()

# Display detector stages and weak classifiers per stage
print('Number of stages: {}'.format(len(cascade_det.strong_CF_list)))
for i in range(len(cascade_det.strong_CF_list)):
    print('Stage {} has {} weak classifiers'.format(i+1, \
          len(cascade_det.strong_CF_list[i].weak_classifiers)))
print('False Positive rate at each stage: {}'.format(Fi_track[1:]))
print('Detection rate at each stage: {}'.format(Di_track[1:]))