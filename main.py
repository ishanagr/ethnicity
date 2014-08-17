#!/usr/bin/python

import sys
import re
import options
import pandas as pd
import numpy as np
from utils import *

(opt, args) = options.argsOptions()

name='AGRAWAL'
category='asian'
folder = '/home/vagrant/ethnicity/'
training_data = 'data/train_header.csv'

name = opt.name
train = opt.train
datafile = opt.datafile
#print('\nName: ' + str(name))
#print('Train: ' + str(train))
#print('Data File: ' + str(datafile) + '\n')

datafile = folder + training_data
#features = {'name':name,'category':category}

### if name is not None: make_prediction(name)
### currently this only prints out the features of the given name
if name is not None: 
    #print('\n- Prediction Mode -\n')
    print('\nSurname: '+name)
    name_dict = generate_features(name)
    category,probabilities = make_prediction(name_dict)
    print('\nEthnicity: '+str(category))
    print('\nProbabilities: '+str(probabilities)+'\n')

### if train is True then train model
### in the future might want to specify training data file
#if train and datafile is None: print('You must specify a training data file!')
#elif train and datafile is not None: 
if train:
    ### first create features
    print('Creating features!')
    #data = read_data(folder+datafile)
    #create_features(data)
    data = pd.read_csv(datafile,header=0) ## get list of names from training data file
    names = data['name'][:] ## extract the list of names from data frame
    create_f(names) ## create features.csv from list of names
    print('Feature creation completed!')

    ### next train models
    determine_model(data,'features.csv')












## TO DO ##
## write features.csv file to data/features.csv, not src/
