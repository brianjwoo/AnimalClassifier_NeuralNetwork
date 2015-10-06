import sys
from utils import *
import cPickle as pickle


def predict_one(img_file_path, show = False):
    img = skimage.io.imread(img_file_path)
    if show:
        skimage.io.imshow(img)
    img_standardized = img_size_standarizer(img, 106, False)
    img_nn = np.array([img_standardized]).astype('float32')
    prediction = nn.predict(img_nn)
    prediction_proba = nn.predict_proba(img_nn)
    return (prediction[0], np.max(prediction_proba))

if __name__ == '__main__':
with open('nn.pkl') as f:
    nn = pickle.load(f)

print predict_one(sys.argv[1])