#!/usr/bin/python

from optparse import OptionParser

def argsOptions():
	usage = "usage: db [query] [options]" 
	parser = OptionParser(usage=usage)
	parser.add_option("-t", "--train", action="store_true", dest="train", default=False, help="use option to train new linguistic model, requires training data")
	parser.add_option("-d", "--data", action="store", dest="datafile", default=None, help="training data in a csv file like: name,category")
	parser.add_option("-n", "--name", action="store", dest="name", default=None, help="name to generate features for")

	(options,args) = parser.parse_args()
	return (options,args)













