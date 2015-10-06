import pandas as pd
import numpy as np
import sklearn.preprocessing

import skimage
import skimage.io
import skimage.color
import skimage.transform

import theano

import nolearn.lasagne
from lasagne import layers
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet, BatchIterator

import cPickle as pickle

#%matplotlib inline


def img_size_standarizer(img, output_size = 128, grey = True):
    '''
    Using skimage.crop, this function takes an img and crops out the larger areas, then the images are resized to
    an appropriate dimension for NN processing. We assume the image is a 3D array (color img)
    '''
    
    d1, d2, c = img.shape
    
    crop = abs(d2-d1)/2
    
    if d1 > d2:
        new_img = skimage.util.crop(img, ((crop,crop),(0,0), (0,0)))
    else:
        new_img = skimage.util.crop(img, ((0,0), (crop,crop), (0,0)))

    
    resize_col_img = skimage.transform.resize(new_img, output_shape=(output_size, output_size, 3))
    
    if grey:
        return skimage.color.rgb2gray(resize_col_img)
    else:
        return np.swapaxes(np.swapaxes(resize_col_img, 1, 2), 0, 1)
    

def preprocessing(img):
    return skimage.color.rgb2gray(img_size_standarizer(img, 96))

def float32(k):
    return np.cast['float32'](k)

class FlipBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        return Xb, yb

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

def predict_one(img_matrix, nn):
    img_standardized = img_size_standarizer(img_matrix, 106, False)
    img_nn = np.array([img_standardized]).astype('float32')
    prediction = nn.predict(img_nn)
    prediction_proba = nn.predict_proba(img_nn)
    if prediction[0] == 0:
        return 'cat', np.max(prediction_proba[0])
    else:
        return 'dog', np.max(prediction_proba[0])



