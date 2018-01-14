from collections import namedtuple
import numpy as np

VggLayer = namedtuple("VggLayer", ['name', 'weight', 'bias'])

def load_weights(file_name):
    weights_npz = np.load(file_name)
    layer_nums = set()
    for file in weights_npz.files:
        layer_nums.add(file[4:7])
    layer_nums = sorted(layer_nums)
    layers = []
    for lnum in layer_nums:
        w = weights_npz['conv'+lnum+'_weight']
        b = weights_npz['conv'+lnum+'_bias']
        # Convert convolutions from Caffe to TF order
        w = w.transpose((2, 3, 1, 0))
        layers.append(VggLayer("conv"+lnum, w, b))
    return layers
