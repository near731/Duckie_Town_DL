# Imports

import numpy as np
from PIL import Image

size=(80, 60)
input_shape=(80, 40, 3)

def preprocess(x, size=size):
  
  x = Image.fromarray(x)
  
  threshold = 175
  x=x.resize(size)
  x=x.crop((0, size[1]/3, size[0], size[1]))
  x=x.point(lambda p: p > threshold and 255)
  return x
