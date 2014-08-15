from flask import Flask, request
app = Flask(__name__)
app.debug = True

import database
import operator
import json
from models import *

def run_model():
	return {'ethnicity': 'unkown', 'probability': 'unknown'};

@app.route('/')
def hello_world():
	return 'Hello World!'

@app.route('/ethnicity')
def ethnicity_search():
	first_name = request.args.get('first_name')
	last_name = request.args.get('last_name')
	result = db.session.query(Name).filter_by(lastname=last_name).all()
	if len(result) != 0:
		ethnicitage = {'hispanic':result[0].hispanic, 'asian':result[0].asian, 'white':result[0].white, 'african':result[0].african}
		print ethnicitage
		top = sorted(ethnicitage.iteritems(), key=operator.itemgetter(1))
		top.reverse()
		res = {'ethnicity': top[0][0], 'probability': top[0][1]};
	else:
		res = run_model()
	#check in names db and return the result, otherwise call other function
	return json.dumps(res)

if __name__ == '__main__':
    app.run()
