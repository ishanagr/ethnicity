#!/usr/bin python

import re
import csv
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn import svm
from sklearn.externals import joblib

folder = '/home/vagrant/ethnicity/'

consonants = list('BCDFGHJKLMNPQRSTVWXYZ')
vowels = list('AEIOU')

def load_patterns(file_name):
    with open(file_name) as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content]
    return(content)

def consonants_vowels(name):
    con_vow = ''
    for i in range(0,len(name)):
        if name[i] in vowels:
            con_vow += 'V'
        if name[i] in consonants:
            con_vow += 'C'
    return(con_vow)

def add_feature(dict,(key,value)):
    dict[key] = value
    return(dict)

##### 
def length(name):
    key = 'length'
    value = len(name)
    return(key,value)

def num_consonants(cv_name):
    key = 'num_consonants'
    value = cv_name.count('C')
    return(key,value)

def num_vowels(cv_name):
    key = 'num_vowels'
    value = cv_name.count('V')
    return(key,value)

def cv_ratio(cv_name):
    key = 'cv_ratio'
    consonants = cv_name.count('C')
    vowels = cv_name.count('V')
    if vowels == 0: vowels = 0.1
    value = round(1.0*consonants/vowels,2)
    return(key,value)

def double_letter_consonant(name):
    key = 'double_letter_consonant'
    if len(re.findall(r'([B-DF-HJ-NP-TV-Z])\1',name))>0: value = 1
    else: value = 0
    return(key,value)

def double_letter_vowel(name):
    key = 'double_letter_vowel'
    if len(re.findall(r'([AEIOU])\1',name))>0: value = 1
    else: value = 0
    return(key,value)

def check_pattern(name,pattern):
    key = pattern
    if pattern in name: value = 1
    else: value = 0
    return(key,value)

def final_letter(name,letter):
    key = letter
    if name.endswith(letter): value = 1
    else: value = 0
    return(key,value)
######

patterns_cv = load_patterns(folder+'data/cv.csv')
patterns_2_letters = load_patterns(folder+'data/letters_2.csv')
patterns_3_letters = load_patterns(folder+'data/letters_3.csv')

######

def generate_features(name):
	#features = {'name':name}
	features = {}
	cv_name = consonants_vowels(name) ## need this to calculate some of the features
	features = add_feature(features, length(name))
	features = add_feature(features, num_consonants(cv_name))
	features = add_feature(features, num_vowels(cv_name))
	features = add_feature(features, cv_ratio(cv_name))
	features = add_feature(features, double_letter_consonant(name))
	features = add_feature(features, double_letter_vowel(name))
	for letter in (consonants + vowels):
	    features = add_feature(features, final_letter(name,letter))  
	for pattern in patterns_cv:
	    features = add_feature(features, check_pattern(cv_name,pattern))  
	for pattern in patterns_2_letters:
	    features = add_feature(features, check_pattern(name,pattern))  
	for pattern in patterns_3_letters:
	    features = add_feature(features, check_pattern(name,pattern))  
	return(features)

def read_data(file_name):
    with open(file_name) as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content]	
    return(content)

def write_features(dict): ## deprecated
	with open('mycsvfile.csv','wb') as f:
	    w = csv.writer(f)
	    w.writerow(dict.keys())
	    w.writerow(dict.values())

def write_f(dict,count):
	if count == 1:
		with open('features.csv', 'wb') as f:  
		    w = csv.DictWriter(f, dict.keys())
		    w.writeheader()
		    w.writerow(dict)
	else: 
		with open('features.csv', 'a') as f:  
		    w = csv.DictWriter(f, dict.keys())
		    w.writerow(dict)

def create_features(training_data): ## deprecated
	count = 1
	for sample in training_data:
		name = sample.split(',')[0]
		#category = sample.split(',')[1]
		name_dict = generate_features(name)
		write_f(name_dict,count)
		count += 1

def create_f(names):
	count = 1
	for name in names:
		name_dict = generate_features(name)
		write_f(name_dict,count)
		count += 1

def determine_model(training_data,features_file):
	df = pd.read_csv(features_file,header=0)
	df['name'] = training_data['name']
	df['category'] = training_data['category']
	df['is_train'] = np.random.uniform(0,1,len(df)) <= 0.75
	train = df[df['is_train']==True]
	test = df[df['is_train']==False]
	train_y = np.array(train['category'])
	test_y = np.array(test['category'])
	features = df.columns[0:403] ## columns you want to use as X's

	print('\nGaussianNB')
	model = GaussianNB() ## call Gaussian Naive Bayesian class with default parameters
	y_hat = model.fit(train[features], train_y).predict(test[features])
	match = (y_hat==test_y)
	result = round(1.0 * sum(match) / len(match),2)
	print('accuracy: '+str(result))
	print('Out of %d points, %d were incorrectly classified.' % (len(test_y),(y_hat != test_y).sum()))

	print('\nSVM')
	model = svm.SVC()
	y_hat = model.fit(train[features], train_y).predict(test[features])
	match = (y_hat==test_y)
	result = round(1.0 * sum(match) / len(match),2)
	print('Accuracy: '+str(result))
	print('Out of %d points, %d were incorrectly classified.' % (len(test_y),(y_hat != test_y).sum()))

	print('\nMultinomialNb')
	model = MultinomialNB()
	y_hat = model.fit(train[features], train_y).predict(test[features])
	match = (y_hat==test_y)
	result = round(1.0 * sum(match) / len(match),2)
	print('Accuracy: '+str(result))
	print('Out of %d points, %d were incorrectly classified.' % (len(test_y),(y_hat != test_y).sum()))

	print('\nRandomForestClassifier')
	model = RandomForestClassifier()
	y_hat = model.fit(train[features], train_y).predict(test[features])
	match = (y_hat==test_y)
	result = round(1.0 * sum(match) / len(match),2)
	print('Accuracy: '+str(result))
	print('Out of %d points, %d were incorrectly classified.' % (len(test_y),(y_hat != test_y).sum()))

	print('\nExtraTreesClassifier')
	model = ExtraTreesClassifier()
	y_hat = model.fit(train[features], train_y).predict(test[features])
	match = (y_hat==test_y)
	result = round(1.0 * sum(match) / len(match),2)
	print('Accuracy: '+str(result))
	print('Out of %d points, %d were incorrectly classified.' % (len(test_y),(y_hat != test_y).sum()))

	print('\nGradientBoostingClassifier')
	model = GradientBoostingClassifier()
	y_hat = model.fit(train[features], train_y).predict(test[features])
	match = (y_hat==test_y)
	result = round(1.0 * sum(match) / len(match),2)
	print('Accuracy: '+str(result))
	print('Out of %d points, %d were incorrectly classified.' % (len(test_y),(y_hat != test_y).sum()))
#####
	fitted_classifier = model.fit(df[features], np.array(df['category']))
	joblib.dump(fitted_classifier, 'classifier.pkl', compress=9)
#####

	print('\nAdaBoostClassifier')
	model = AdaBoostClassifier()
	y_hat = model.fit(train[features], train_y).predict(test[features])
	match = (y_hat==test_y)
	result = round(1.0 * sum(match) / len(match),2)
	print('Accuracy: '+str(result))
	print('Out of %d points, %d were incorrectly classified.' % (len(test_y),(y_hat != test_y).sum()))

	print('\nLinear SVM')
	model = svm.LinearSVC()
	y_hat = model.fit(train[features], train_y).predict(test[features])
	match = (y_hat==test_y)
	result = round(1.0 * sum(match) / len(match),2)
	print('Accuracy: '+str(result))
	print('Out of %d points, %d were incorrectly classified.' % (len(test_y),(y_hat != test_y).sum()))

	print('\nRBF SVM')
	model = svm.SVC(kernel='rbf')
	y_hat = model.fit(train[features], train_y).predict(test[features])
	match = (y_hat==test_y)
	result = round(1.0 * sum(match) / len(match),2)
	print('Accuracy: '+str(result))
	print('Out of %d points, %d were incorrectly classified.' % (len(test_y),(y_hat != test_y).sum()))

	print('\nDecisionTreeClassifier')
	model = DecisionTreeClassifier()
	y_hat = model.fit(train[features], train_y).predict(test[features])
	match = (y_hat==test_y)
	result = round(1.0 * sum(match) / len(match),2)
	print('Accuracy: '+str(result))
	print('Out of %d points, %d were incorrectly classified.' % (len(test_y),(y_hat != test_y).sum()))

	print('\nRidgeClassifierCV')
	model = RidgeClassifierCV()
	y_hat = model.fit(train[features], train_y).predict(test[features])
	match = (y_hat==test_y)
	result = round(1.0 * sum(match) / len(match),2)
	print('Accuracy: '+str(result))
	print('Out of %d points, %d were incorrectly classified.' % (len(test_y),(y_hat != test_y).sum()))

def make_prediction(name_dict):
	features = np.array(name_dict.items())[:,1].astype(float)
	reloaded = joblib.load('classifier.pkl')
	prediction = reloaded.predict(features)
	probabilities = reloaded.predict_proba(features)
	return(prediction_to_category(prediction),probas_to_categories(probabilities))

def prediction_to_category(factor):
	if factor == 1: return('Asian')
	if factor == 2: return('Hispanic')
	if factor == 3: return('White')

def probas_to_categories(probas):
	category_probas = {'Asian':round(probas[0][0],2),'Hispanic':round(probas[0][1],2),'White':round(probas[0][2],2)}
	return(category_probas)







