# -*- coding: utf-8 -*-

# Author: Gopikrishnan Sasi Kumar

# This script saves the train data to file saved_data_train.npz

import library_adaboost_hw10 as lib
lib.extract_and_save('train/positive', 'train/negative', 'saved_data_train')