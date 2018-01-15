# Style transfer in TensorFlow
This repository contains an implementation of "A Neural Algorithm of
Artistic Style" by Gatys et al
([arXiv](https://arxiv.org/abs/1508.06576)).

Once installed, the program can be run on two input images to produce
a third re-styled image:
```
./styletransfer -n 100 -c 1e-3 image.jpg art.jpg out.jpg
```
The available options are printed when run as `./styletransfer -h`.

## Installation
This program depends on [TensorFlow](https://www.tensorflow.org/) and
[Pillow](http://python-pillow.github.io/). Install these first.

Next, obtain the VGG network weights. Download the `vgg16-weights.npz`
file and extract it if necessary (if it was distributed gzip
compressed as a `.npz.gz` file).

Finally, clone the repository or otherwise obtain the source code and
place the `.npz` file next to the `styletransfer` program.

## License
The source code in this repository is distributed under the MIT
license. See `LICENSE.txt` for the license text.

### VGG Weights
The VGG network weights were extracted from the original files
distributed by the Visual Geometry Group at Oxford. The original files
can be obtained from the group's
[website](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) and are
distributed under the terms of the Creative Commons Attribution
License ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

Details on the networks are available in the group's technical report
([arXiv](https://arxiv.org/abs/1409.1556.pdf)).
