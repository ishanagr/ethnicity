from flask import Flask, request, render_template, url_for
app = Flask(__name__)
app.debug = True

import database
import operator
import json
import pandas as pd
import numpy as np
from models import *
from utils import *

def run_model(name):
	name_dict = generate_features(name)
	#print(name_dict)
	category,probabilities = make_prediction(name_dict)
	#print(category)
	#print(probabilities)
	key=max(probabilities.iterkeys(), key=(lambda key: probabilities[key]))
	probability = round(100*probabilities[key])
	percent = str(int(probability))+'%' 
	return {'ethnicity': category, 'probability': percent};

@app.route('/ethnicity', methods=['GET', 'POST'])
def ethnicity():
	if request.method == 'GET':
		return render_template('form.html')
	else:
		name=request.form['lastname']
		res = ethnicity_search(name)
		return render_template('result.html', ethnicity=res['ethnicity'], probability=res['probability'])

@app.route('/ethnicityapi')
def ethnicityapi():
	first_name = request.args.get('first_name')
	last_name = request.args.get('last_name')
	return json.dumps(ethnicity_search(last_name))

def ethnicity_search(last_name):
	result = db.session.query(Name).filter_by(lastname=last_name).all()
	if len(result) != 0:
		ethnicitage = {'hispanic':result[0].hispanic, 'asian':result[0].asian, 'white':result[0].white, 'african':result[0].african}
		top = sorted(ethnicitage.iteritems(), key=operator.itemgetter(1))
		top.reverse()
		res = {'ethnicity': top[0][0], 'probability': str(int(top[0][1]))+'%'};
	else:
		res = run_model(last_name)
	#check in names db and return the result, otherwise call other function
	return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
