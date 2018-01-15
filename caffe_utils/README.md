# VGG Weight Extraction
The weights for the VGG16 and VGG19 models are distributed in
`.caffemodel` format. The Python script in this directory can extract
the convolution kernel and bias values and store them in an `.npz`
file suitable for reading with [NumPy](http://www.numpy.org/).

The original `.caffemodel` files can be obtained from the Visual
Geometry Group's
[website](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

## Extracted Format
The extracted values are stored in the `.npz` file with names
corresponding to the VGG network layer from which they were drawn.

For example the `conv1_1` layer is stored in records `conv1_1_weight`
for the convolution kernel values and `conv1_1_bias`.

The convolution kernel values are stored in *Caffe*
[order](http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html#blob-storage-and-communication).
These need to be converted to TensorFlow
[order](https://www.tensorflow.org/api_docs/python/tf/nn/convolution). In
NumPy, this conversion can be performed during loading with:
```
weight.transpose((2, 3, 1, 0))
```
where `weight` is one of the convolution kernel values. The bias
values require no conversion.

## Installation
This script depends on [NumPy](http://www.numpy.org/) and on Google's
[Protocol Buffers](https://github.com/google/protobuf) library for
Python.

This script also requires a compiled version of the `caffe.proto` file
from the Caffe project. Obtain the source definitions from the
repository
[here](https://github.com/BVLC/caffe/blob/2cbc1bba0922c29241e2474dd43b180be265229f/src/caffe/proto/caffe.proto)
then compile them with:
```
protoc --python_out=. caffe.proto
```
This will produce the `caffe_pb2.py` file needed by this script. Place
it next to `caffe_convert.py`

## Running
Run the script by specifying first the input `.caffemodel` file
followed by the destination `.npz` file name. For example:
```
./caffe_convert.py VGG_ILSVRC_16_layers.caffemodel vgg16-weights.npz
```
The process is quite slow and may take a long time to run.
