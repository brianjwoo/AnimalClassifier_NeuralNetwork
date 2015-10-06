import os
import sys
import random
import pandas as pd
from flask import Flask, render_template, request
from image_request import *
from utils import *

app = Flask(__name__)

@app.route('/index.html')
@app.route('/', methods = ['GET', 'POST'])
def post_index(url = None, img_name = None):
    random.shuffle(filler_imgs)
    if request.method == 'POST':
        url = request.form['url']
        print url
        try:
    	    img_name, img_matrix = import_img(url)
            y_pred, y_pred_proba = predict_one(img_matrix)
        except:
            img_name = 'Error'
    #Download img @ url into a specific folder
    #open a database connecition loading incorrectly named photos
    #generate a dictionary using the connection
    #load information target classified and probability as well as preivously rated 
    #
    return render_template('index.html', img_name=img_name, instances = filler_imgs[:4])



@app.route('/about.html')
def about():
	return render_template('about.html')

@app.route('/contact.html')
def contact():
	return render_template('contact.html')



if __name__ == '__main__':
	#prepare a py init folder
	#load utilities for prediction
    with open('nn.pkl') as f:
        nn = pickle.load(f)
	#take image url
	#download image and store
	#store string in posgres
	#predict

	#CHECK IF TABLE EXISTS OTHERwiSE POPULATE

	#generate incorrect labels

    q = '''
	SELECT * FROM __ WHERE y_pred NOT EQUAL 

	'''
    filler_imgs = os.listdir('./static/data/new_images/')

    app.run(debug=True)
    #app.run(host = '0.0.0.0')#, port=80, debug = True)

