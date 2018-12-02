# -*- coding: utf-8 -*-

# Author: Gopikrishnan Sasi Kumar

# This script tests the cascaded detector saved in 
# trained_cascaded_detector.pkl using the classical threshold

import pickle
import library_adaboost_hw10 as lib
import matplotlib.pyplot as plt
dpi_value = 600

with open('trained_cascaded_detector.pkl', 'rb') as input_file:
    Fi_track = pickle.load(input_file)
    Di_track = pickle.load(input_file)
    cascade_det = pickle.load(input_file)

# Load test images
images_test_pos = lib.load_images('test/positive')
images_test_neg = lib.load_images('test/negative')

strong_classifiers = cascade_det.strong_CF_list
FP_all = []
FN_all = []

for i in range(len(strong_classifiers)):
    new_detector = lib.cascaded_detector(strong_classifiers[:i+1])
    [FP, FN] = lib.test_cascade_detector_classical(images_test_pos, \
                                        images_test_neg,\
                                        new_detector)
    FP_all.append(FP)
    FN_all.append(FN)

x_ax = list(range(1, len(FP_all) + 1))
fig = plt.figure()
plt.plot(x_ax, FP_all, color='red', marker='x', markersize = 5,\
         label = 'False Positive Rate')
plt.plot(x_ax, FN_all, color='blue', marker='x', markersize = 5,\
         label = 'False Negative Rate')
plt.xlabel('Cascade Stage')
plt.ylabel('Rate')
plt.xticks(x_ax)
plt.legend()
plt.savefig('Cascade_detector_test_rates_classical.png', dpi = dpi_value)
plt.close()