
import os
import sys
import random
import pandas as pd
from flask import Flask, render_template, request
from flask.ext.sqlalchemy import SQLAlchemy

from image_request import *
from utils import *

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
db = SQLAlchemy(app)

class AnimalImg(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.String(30))
    url = db.Column(db.String(80), unique=True)
    classification = db.Column(db.String(10))
    probability = db.Column(db.Float)
    dog_count = db.Column(db.Integer)
    cat_count = db.Column(db.Integer) 

    def __init__(self, url, classification, probability):
        self.url = url
        self.image = url.split('/')[-1]
        self.classification = classification
        self.probability = probability
        self.dog_count = 0
        self.cat_count = 0

    def __repr__(self):
        return self.url


@app.route('/index.html')
@app.route('/', methods = ['GET', 'POST'])
def index(url = None, img_name = None):
    prediction = (None, None)
    #random.shuffle(filler_imgs) #This is temporary for bottom portion
    if request.method == 'POST':
        url = request.form['url']
#        print url
        try:
    	    img_name, img_matrix = import_img(url)
#            print img_name, img_matrix.shape
            prediction = predict_one(img_matrix, nn) #Tuple (animal and probability)
            db.session.add(AnimalImg(url, prediction[0], prediction[1]))
            db.session.commit() 
        except:
            img_name = 'Error'
    #Download img @ url into a specific folder
    #open a database connecition loading incorrectly named photos
    #generate a dictionary using the connection
    #load information target classified and probability as well as preivously rated 
    animal_instances = AnimalImg.query.limit(4)

    return render_template('index.html', img_name=img_name, prediction = prediction, instances = animal_instances)



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

    filler_imgs = os.listdir('./static/data/new_images/')

    #app.run(debug=True)
    app.run(host = '0.0.0.0', port=5000, debug = True)

