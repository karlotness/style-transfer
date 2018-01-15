#! /usr/bin/env python3
# Copyright (c) 2018 Karl Otness
import sys
import numpy as np
import caffe_pb2 as caffe

def convert_layer(layer):
    sz = layer.convolution_param.kernel_size[0]
    n_out = layer.convolution_param.num_output
    weight = np.array(layer.blobs[0].data).reshape(n_out, -1, sz, sz)
    bias = np.array(layer.blobs[1].data)
    return weight, bias

if len(sys.argv) < 3:
    print("Usage: {} in_caffe_model out_npz".format(sys.argv[0]))
    exit(1)

with open(sys.argv[1], 'rb') as model_file:
    file_data = model_file.read()
net_param = caffe.NetParameter()
net_param.ParseFromString(file_data)
del file_data

layers = {}
for i in range(len(net_param.layers)):
    name = net_param.layers[i].name
    if not name.startswith('conv'):
        # Only extract convolution layers
        continue
    weight, bias = convert_layer(net_param.layers[i])
    layers[name+'_weight'] = weight
    layers[name+'_bias'] = bias

np.savez(sys.argv[2], **layers)
