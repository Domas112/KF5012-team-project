from PIL import Image
import numpy as np

def preprocess(img):

    img = img.convert('L')
    img = np.array(img.resize((256,256)))
    img = img.reshape((256,256,1))
    
    return img