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

#Read Data into Python
cats_collection = skimage.io.imread_collection('./data/train/cat*')
dogs_collection = skimage.io.imread_collection('./data/train/dog*')

#Generate input matrix
X = []
y = []

for c in cats_collection:
    X.append(img_size_standarizer(c, 106, False))
    y.append(0)
for d in dogs_collection:
    X.append(img_size_standarizer(d, 106, False))
    y.append(1)
    
X = np.array(X).astype('float32')
y = np.array(y).astype('int32')

np.random.seed(42)
index = np.random.permutation(len(X))

X = X[index]
y = y[index]

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


nn = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout0', layers.DropoutLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer), 
        ('hidden5', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer), 
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, X.shape[1], X.shape[2], X.shape[3]),
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout0_p = .3,
    conv3_num_filters=256, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout1_p = .4,
    hidden4_num_units=1024, dropout2_p = .5,
    hidden5_num_units=512, dropout3_p = .5,
    output_num_units=2, output_nonlinearity=softmax,

    batch_iterator_train=FlipBatchIterator(batch_size=256),

    update_learning_rate=theano.shared(float32(.01)),
    update_momentum=theano.shared(float32(.9)),

    regression=False,

    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.01, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=20),
        ],

    max_epochs=1000,
    verbose=1,
    )

nn.fit(X,y)

#y_pred = nn.predict(X)

#for y, y_pred in zip(y,y_pred):
#    print y, y_pred

with open('nn.pkl', 'wb') as f:
    pickle.dump(nn, f, -1)
