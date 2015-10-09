
import os
import sys
import random
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from flask.ext.sqlalchemy import SQLAlchemy
from sqlalchemy.sql.expression import func

from image_request import *
from utils import *

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./animal_img.db'
db = SQLAlchemy(app)

class AnimalImg(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.String(30))
    url = db.Column(db.String(80), unique=True)
    classification = db.Column(db.String(10))
    probability = db.Column(db.Float)
    dog_count = db.Column(db.Integer)
    cat_count = db.Column(db.Integer)
    unknown_count = db.Column(db.Integer) 

    def __init__(self, url, classification, probability):
        self.url = url
        self.image = url.split('/')[-1]
        self.classification = classification.capitalize()
        self.probability = round(probability, 3)
        self.dog_count = 0
        self.cat_count = 0
        self.unknown_count = 0

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
    animal_instances = AnimalImg.query.order_by(func.random()).limit(4)
    #animal_instances = AnimalImg.query.limit(4)
    #animal_instances = db.session.query(AnimalImg).order_by(func.random()).limit(4)
    return render_template('index.html', img_name=img_name, prediction = prediction, instances = animal_instances)

@app.route('/<image>/<classification>')
def update(image = None, classification = None):
	animal_update = AnimalImg.query.filter_by(image=image).first()
 	#print animal_update
	if classification == 'cat':
		animal_update.cat_count +=1
	elif classification == 'dog':
		animal_update.dog_count +=1
	else:
		animal_update.unknown_count +=1
	db.session.commit()
	return redirect(url_for('index'))


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

    #filler_imgs = os.listdir('./static/data/new_images/')

    #app.run(debug=True)
    app.run(host = '0.0.0.0', port=5000, debug = True)

