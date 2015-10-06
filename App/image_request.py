
import sys
import requests
import pandas as pd
from PIL import Image
from StringIO import StringIO

def import_img(img_url):
'''
Takes an img url and returns the file name and img matrix
'''
    r = requests.get(img_url)
    i = Image.open(StringIO(r.content))
    i.save('./static/data/new_images/' + img_url.split('/')[-1])
    return img_url.split('/')[-1], np.asarray(i)
if __name__ == '__main__':
    print sys.argv[1]
    import_img(sys.argv[1])