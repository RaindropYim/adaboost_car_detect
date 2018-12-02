# -*- coding: utf-8 -*-

# Author: Gopikrishnan Sasi Kumar

# This library contains all the functions used in the Adaboost classifier 
# task of Homework 10 of ECE661.

import os
import cv2
import numpy as np
import pickle

class feature():
    # Defines the feature
    def __init__(self, ftype, corner_dim):
        # Feature is defined by:
        # 1. ftype: A, B, C, D
        # 2. corners: [x_0, y_0, W, H] of first box
        self.ftype = ftype
        self.x0 = corner_dim[0]
        self.y0 = corner_dim[1]
        self.W = corner_dim[2]
        self.H = corner_dim[3]
        
    
    def evaluate(self, iimage):
        # Evaluate the feature for an integral image
        if self.ftype == 'A':
            plus = self.area_eval(iimage, [self.x0, self.y0],\
                                  self.W, self.H)
            minus = self.area_eval(iimage, [self.x0 + self.W, self.y0],\
                                  self.W, self.H)
            return plus - minus
        elif self.ftype == 'B':
            minus = self.area_eval(iimage, [self.x0, self.y0],\
                                  self.W, self.H)
            plus = self.area_eval(iimage, [self.x0, self.y0 + self.H],\
                                  self.W, self.H)
            return plus - minus
        elif self.ftype == 'C':
            plus = self.area_eval(iimage, [self.x0, self.y0],\
                                  self.W, self.H)
            minus = self.area_eval(iimage, [self.x0 + self.W, self.y0],\
                                  self.W, self.H)
            plus += self.area_eval(iimage, [self.x0 + 2 * self.W, self.y0],\
                                  self.W, self.H)
            return plus - minus
        elif self.ftype == 'D':
            plus = self.area_eval(iimage, [self.x0, self.y0],\
                                  self.W, self.H)
            minus = self.area_eval(iimage, [self.x0 + self.W, self.y0],\
                                  self.W, self.H)
            minus += self.area_eval(iimage, [self.x0, self.y0 + self.H],\
                                  self.W, self.H)
            plus += self.area_eval(iimage, [self.x0 + self.W,\
                                            self.y0 + self.H],\
                                  self.W, self.H)
            return plus - minus
            
            
    def area_eval(self, iimage, pivot, width, height):
        # Evaluate the pixel intensity within bounded box in integral image
        # iimage
        # Pivot = (x,y)
        col, row = pivot
        one = iimage[row, col]
        two = iimage[row, col + width]
        three = iimage[row + height, col]
        four = iimage[row + height, col + width]
        return four + one - (two + three)


class weak_classifier():
    # Defines the weak classifier
    def __init__(self, f, theta, p):
        # Weak classifier is defined by:
        # 1. feature f (f belongs to class 'feature')
        # 2. threshold theta
        # 3. polarity p
        self.f = f
        self.theta = theta
        self.p = p
    
    
    def classify(self, iimage):
        # Classify an integral image using the classifier
        fx = self.f.evaluate(iimage)
        if self.p * fx < self.p * self.theta:
            return 1
        else:
            return 0
        

class strong_classifier():
    # Defines the strong classifier
    def __init__(self, weak_class_list, alphas, threshold):
        # Strong classifier is defined by a list of weak classifiers
        self.weak_classifiers = weak_class_list
        self.alphas = alphas
        self.threshold = threshold
        
        
    def classify_classical(self, iimage):
        # Classify an integral image uwithout using the threshold
        lhs = 0
        rhs = 0
        for i in range(len(self.alphas)):
            lhs += self.alphas[i] * self.weak_classifiers[i].classify(iimage)
            rhs += self.alphas[i]
        rhs = rhs/2
        if lhs >= rhs:
            return 1
        else:
            return 0
        
    def classify_threshold(self, iimage):
        # Classify an integral image using the threshold
        lhs = 0
        for i in range(len(self.alphas)):
            lhs += self.alphas[i] * self.weak_classifiers[i].classify(iimage)
        if lhs >= self.threshold:
            return 1
        else:
            return 0
        

class cascaded_detector():
    # This class defines the cascaded detector
    def __init__(self, strong_CF_list):
        self.strong_CF_list = strong_CF_list
        
    def classify(self, iimage):
        for strong_CF in self.strong_CF_list:
            # If either one of the stages outputs 0, return 0
            ynext = strong_CF.classify_threshold(iimage)
            if ynext == 0:
                return 0
        return 1
    

def load_images(foldername):
    # Load all the images in the folder to a list
    images = []
    for root, dirs, files in os.walk(foldername, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            # load images in grayscale
            image = cv2.imread(filename, 0)
            images.append(cv2.integral(image))
    return images


def generate_features():
    # Generate the set of all features to be used
    features = []
    # Type A features
    for h in range(1, 21):
        for w in range(1, 21):
            for y0 in range(21-h):
                for x0 in range(41-2*w):
                    features.append(feature('A', [x0, y0, w, h]))
    
    for h in range(1, 11):
        for w in range(1, 41):
            for y0 in range(21-2*h):
                for x0 in range(41-w):
                    features.append(feature('B', [x0, y0, w, h]))
    print('Total number of features = {}'.format(len(features)))
    return features


def extract_features(images, features):
    # Extracts features from images 
    extracted_features = []
    count = 0
    for feature in features:
        values = []
        for image in images:
            val = feature.evaluate(image)
            values.append(val)
        values = np.array(values)
        ind = np.argsort(values)
        values = values[ind]
        extracted_features.append(np.c_[values, ind])
        count += 1
        if count % 10000 == 0:
            print('Features processed: {} of {}'.format(count, \
                  len(features)))
    return extracted_features


def extract_and_save(pos_images_folder, neg_images_folder, filename):
    # load train data
    images_train_pos = load_images(pos_images_folder)
    images_train_neg = load_images(neg_images_folder)
    
    # Combine the positive and negative images
    y_pos = np.ones((len(images_train_pos),))
    y_neg = np.zeros((len(images_train_neg),))
    yi_train = np.concatenate((y_pos, y_neg))
    images_train = images_train_pos
    images_train.extend(images_train_neg)
    print(yi_train.shape, len(images_train))
    
    # Generate features
    features = generate_features()
    
    # Extract features from images
    features_train = extract_features(images_train, features)
    # this is a list with each element corresponding to each of the features
    
    np.savez_compressed(filename, features_train=features_train, \
             yi_train=yi_train)
    
    return 1


def load_data(filename):
    # Load the saved data from file
    npzfile = np.load(filename)
    features_train = npzfile['features_train']
    yi_train = npzfile['yi_train']
    return [features_train, yi_train]


def find_best_weak_classifier(ext_features, yi, weights, T_plus, T_minus):
    # Find the best weak classifier for the given features, labels 
    # and weights
    errors = []
    polarities = []
    thresholds = []
    for feature in ext_features:
        val = feature[:, 0]
        indices = feature[:, 1]
        # rearrange labels and weights based on indices
        labels = yi[indices]
        arranged_weights = weights[indices]
        
        s_plus = 0
        s_minus = 0
        
        e = []
        p = []
        for i in range(len(val)):
            arg1 = s_plus + (T_minus - s_minus)
            arg2 = s_minus + (T_plus - s_plus)
            if arg1 < arg2:
                e.append(arg1)
                p.append(-1)
            else:
                e.append(arg2)
                p.append(1)
            if labels[i] == 1:
                s_plus += arranged_weights[i]
            elif labels[i] == 0:
                s_minus += arranged_weights[i]
            else:
                print('Fatal error! Label not equal to 0 or 1 encountered!')
        arg1 = s_plus + (T_minus - s_minus)
        arg2 = s_minus + (T_plus - s_plus)
        if arg1 < arg2:
            e.append(arg1)
            p.append(-1)
        else:
            e.append(arg2)
            p.append(1)
        e = np.array(e)
        min_arg = np.argmin(e)
        # e has length 1 greater than val. Handle the extreme conditions.
        if min_arg == 0:
            thresh = val[min_arg - 1] - 10
        elif min_arg == len(e)-1:
            thresh = val[min_arg - 1] + 10
        else:
            thresh = (val[min_arg] + val[min_arg-1])/2
        errors.append(e[min_arg])
        polarities.append(p[min_arg])
        thresholds.append(thresh)
    errors = np.array(errors)
    min_arg = np.argmin(errors)
    
    # Figure out which all images are classified correctly
    feature = ext_features[min_arg]
    val = feature[:, 0]
    indices = feature[:, 1]
    sort_ind = np.argsort(indices)
    val = val[sort_ind]
    
    if polarities[min_arg] == -1:
        y = 1 * (val > thresholds[min_arg])
        ei = 1 * (yi == y)
    else:
        y = 1 * (val < thresholds[min_arg])
        ei = 1 * (yi == y)
    
    return [min_arg, \
            thresholds[min_arg], \
            polarities[min_arg], \
            errors[min_arg], ei, y]

    
def adaBoost(ext_feat_train, \
             yi_train, \
             num_positive_images = 710, \
             num_negative_images = 1758, \
             T = 20, \
             d_min = 1, \
             fp_max = 0.5):
    # Carry out adaBoost algorithm for given set of positive and
    # negative images
    # d_min is the min acceptable detection rate per layer
    # fp_max is the max acceptable false positive rate per layer
    
    # Generate feature descriptors
    features = generate_features()
    
    # Initialize weights
    pos_weights = 0.5/num_positive_images * np.ones((num_positive_images, ))
    neg_weights = 0.5/num_negative_images * np.ones((num_negative_images, ))
    weights = np.concatenate((pos_weights, neg_weights))
    
    weak_class_list = []
    alpha_t = []
    alphat_ht_sum = np.zeros((len(weights), ))
    for t in range(T):
        # Normalize weights
        weights = weights / weights.sum()
        
        # Find the best weak classifier
        indices = np.argwhere(yi_train == 1)
        plus_weights = weights[indices]
        T_plus = plus_weights.sum()
        
        indices = np.argwhere(yi_train == 0)
        minus_weights = weights[indices]
        T_minus = minus_weights.sum()
        
        [feat_num, thresh, pol, err, ei, ht] = find_best_weak_classifier(\
        ext_feat_train, yi_train, weights, T_plus, T_minus)
        print('Feature index:{}, thres:{}, p:{}, Wted error:{}'.format(\
              feat_num, thresh, pol, err))
        print('Successrate = {}'.format(ei.sum()/len(ei)))
        
        weak_class_list.append(weak_classifier(features[feat_num], \
                                                 thresh,\
                                                 pol))
        
        # Update weights
        # ei is currently 1 if classified correctly
        beta = err/(1 - err)
        # ei needs to be 0 if classified correctly. Hence ei instead of 1-ei
        weights = weights * (beta ** ei)
        
        # Save alpha for later
        alpha_t.append(np.log(1/beta))
        print('alpha_t = {}'.format(alpha_t[t]))
        
        # Compute error rate for positive examples and false-positive rate
        
        rhs_thresh = 0.5 * sum(alpha_t)
        alphat_ht = alpha_t[t] * ht
        alphat_ht_sum = alphat_ht_sum + alphat_ht
        strong_classifier_output = 1 * (alphat_ht_sum >= rhs_thresh)
        output_pos = strong_classifier_output[:num_positive_images]
        output_neg = strong_classifier_output[num_positive_images:]
        pos_err = output_pos.sum()/num_positive_images
        false_positive = output_neg.sum()/num_negative_images
        
        print('-' * 40)
        print('Strong classifier stats (running)')
        print('Before tuning:')
        print('Detection rate = {}'.format(pos_err))
        print('False positive rate = {}'.format(false_positive))
        print('-' * 40)
        
        # We tune the final classifier threshold to get at least detection 
        # rate of d_min
        if d_min == 1:
            rhs_thresh_new = alphat_ht_sum[:num_positive_images]
            rhs_thresh_new = rhs_thresh_new.min()
            strong_classifier_output = 1 * (alphat_ht_sum >= rhs_thresh_new)
            output_pos = strong_classifier_output[:num_positive_images]
            output_neg = strong_classifier_output[num_positive_images:]
            pos_err = output_pos.sum()/num_positive_images
            false_positive = output_neg.sum()/num_negative_images
        else:
            rhs_thresh_new = rhs_thresh
            while(pos_err < d_min):
                # Go on reducing the threshold until d_min achieved
                rhs_thresh_new -= abs(rhs_thresh)/100
                strong_classifier_output = 1 * (alphat_ht_sum >= rhs_thresh_new)
                output_pos = strong_classifier_output[:num_positive_images]
                output_neg = strong_classifier_output[num_positive_images:]
                pos_err = output_pos.sum()/num_positive_images
                false_positive = output_neg.sum()/num_negative_images
                print(pos_err, false_positive)
        print('After tuning:')
        print('Detection rate = {}'.format(pos_err))
        print('False positive rate = {}'.format(false_positive))
        print('-' * 40)
        # Stop execution if pos_err = 1 and false_positive <= 0.5
        if pos_err >= d_min and false_positive <= fp_max:
            break
    print('Strong classifier stats (Final)')
    print('Detection rate = {}'.format(pos_err))
    print('False positive rate = {}'.format(false_positive))
    print('=' * 40)
    strong_classif = strong_classifier(weak_class_list, alpha_t, \
                                       rhs_thresh_new)
    return [strong_classif, false_positive, pos_err]


def train_cascaded_detector(F_target = 0.0,\
                            saved_data_train = 'saved_data_train.npz', \
                            neg_train_folder = 'train/negative', \
                            num_pos_train = 710, \
                            num_neg_train = 1758, \
                            T=20, \
                            dmin_per_layer=1, \
                            fmax_per_layer=0.5):
    # Train the cascaded detector
    # F_target is the targeted False Positive rate
    # P and N are taken from saved_data_train
    # Test data are taken from saved_data_test
    # Load data saved for train images
    [ext_feat_train, yi_train] = load_data(saved_data_train)
    ext_feat_train = list(ext_feat_train)
    # Comment this
    # ext_feat_train = ext_feat_train[:1000]
    # Comment this ^^
    
    F0 = 1.0
    D0 = 1.0
    Fi = F0  # Current FP rate
    Di = D0  # Current Detection rate
    # ind_to_retain = np.ones((num_pos_train + num_neg_train, ))
    cascade = []
    cascade_det = None
    neg_train_images_tracker = np.arange(num_neg_train, dtype = int)
    Fi_track = [F0]
    Di_track = [D0]
    stage = 0
    while (Fi > F_target):
        stage += 1
        num_neg_train = len(neg_train_images_tracker)  # changes every itern
        [strong_CF, F, D] = adaBoost(ext_feat_train, \
                             yi_train, \
                             num_pos_train, \
                             num_neg_train, \
                             T, \
                             dmin_per_layer, \
                             fmax_per_layer)
        Fi = Fi * F
        Di = Di * D
        cascade.append(strong_CF)
        print('='*10)
        print('Stage {}'.format(stage))
        print('='*10)
        print('Fi = {}, Di = {}'.format(Fi, Di))
        Fi_track.append(Fi)
        Di_track.append(Di)
        cascade_det = cascaded_detector(cascade)

        # Update N with false detections in latest classifier
        if Fi > F_target:
            [ext_feat_train, yi_train, neg_train_images_tracker] = update_neg_data(ext_feat_train, \
                                                        num_pos_train,\
                                                        neg_train_folder,\
                                                        cascade_det,\
                                                    neg_train_images_tracker)
    np.savez('trained_cascaded_detector', \
             Fi_track=Fi_track, \
             Di_track=Di_track, \
             cascade_det=cascade_det)
    with open('trained_cascaded_detector.pkl', 'wb') as output:
        pickle.dump(Fi_track, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(Di_track, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(cascade_det, output, pickle.HIGHEST_PROTOCOL)
    
    return [Fi_track, Di_track, cascade_det]
            
            
def update_neg_data(ext_feat_train, num_pos_train, neg_train_folder, \
                    cascade_det, neg_train_images_tracker):
    # Update negative images based on output of cascaded detector
    images_train_neg = load_images(neg_train_folder)
    # images_train_neg = images_train_neg[neg_train_images_tracker]
    output = [cascade_det.classify(images_train_neg[i])\
              for i in neg_train_images_tracker]
    output = np.array(output, dtype = bool)
    
    neg_train_images_tracker = neg_train_images_tracker[output]
    
    # Take only those images which are classified as 1
    for i in range(len(ext_feat_train)):
        feature = ext_feat_train[i]
        val = feature[:, 0]
        ind = feature[:, 1]
        indices = np.argsort(ind)
        val = val[indices]
        val_pos = val[:num_pos_train]
        val_neg = val[num_pos_train:]
        val_neg = val_neg[output]
        val = np.concatenate((val_pos, val_neg))
        indices = np.argsort(val)
        val = val[indices]
        # print(val.shape, indices.shape)
        new_feature = np.c_[val, indices]
        # print(ext_feat_train.shape)
        ext_feat_train[i] = new_feature
        
    y_pos = np.ones((len(val_pos),))
    y_neg = np.zeros((len(val_neg),))
    yi_train = np.concatenate((y_pos, y_neg))
    
    return [ext_feat_train, yi_train, neg_train_images_tracker]


def test_cascade_detector(images_test_pos, images_test_neg,\
                          cascade_det):
    # Finds the False positive rate and False negative rate for test images
    # Classify test images
    classify_pos = [cascade_det.classify(iimage) for iimage in images_test_pos]
    classify_neg = [cascade_det.classify(iimage) for iimage in images_test_neg]

    # Find False Positive Rate
    FP = sum(classify_neg) / len(classify_neg)
    
    # Find False Negative Rate
    FN = 1 - sum(classify_pos) / len(classify_pos)
    
    return [FP, FN]


def test_cascade_detector_classical(images_test_pos, images_test_neg,\
                          cascade_det):
    # Finds the False positive rate and False negative rate for test images
    # Classify test images
    classify_pos = [classify_classical(cascade_det, iimage) for iimage in images_test_pos]
    classify_neg = [classify_classical(cascade_det, iimage) for iimage in images_test_neg]

    # Find False Positive Rate
    FP = sum(classify_neg) / len(classify_neg)
    
    # Find False Negative Rate
    FN = 1 - sum(classify_pos) / len(classify_pos)
    
    return [FP, FN]


def classify_classical(cascade_det, iimage):
    # Classify iimage using the classical threshold
    for strong_CF in cascade_det.strong_CF_list:
        # If either one of the stages outputs 0, return 0
        ynext = strong_CF.classify_classical(iimage)
        if ynext == 0:
            return 0
    return 1