from collections import namedtuple
from PIL import Image
import numpy as np

def to_bgr(img_arr):
    return img_arr[:, :, ::-1]

def convert_image(img):
    arr = np.array(img).astype('float32')
    mean = arr.mean(axis = (0, 1))
    data = to_bgr(arr - mean)
    return data, mean

def unconvert_image(data, mean):
    rgb = to_bgr(data)
    arr = (rgb + mean).clip(0, 255).astype('uint8')
    return Image.fromarray(arr)
