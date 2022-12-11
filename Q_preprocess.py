# Ez működik
import numpy as np
from PIL import Image
# original size (800, 600)
size=(80, 60)
input_shape=(40, 80, 3)

def preprocess(x, size=size):
  
  x = Image.fromarray(x)
  
  threshold = 175
  x=x.resize(size)
  x=x.crop((0, size[1]/3, size[0], size[1]))
  x=x.point(lambda p: p > threshold and 255)
  x=np.asarray(x) / 255 # 0 or 1
  return x
