# Sketch-A-XNORNet
An implementation of a variation of [Sketch-A-Net] using [XNOR-Net] in [TensorFlow].

**Sketch-A-Net** is a multi-scale multi-channel deep neural network framework that, for the first time, yields sketch recognition performance surpassing that of humans.

**XNOR-Net** is a vairation of standard convolutional neural networks with an approximates convolutions using primaryly binary operations. XNOR-Net approximates the weights and the input tensors to be binary numbers {-1, +1} which allows faster computation allowing 32x memory saving and 58x faster convolution operations.

## Data
I will be using the [TU-Berlin Sketch Dataset], which is the most commonly used human sketch dataset. It contains 250 categories with 80 sketches per category. It was collected on Amazon Mechanical Turk from 1350 participants, thus providing a variety of sketches for each category. The images are available in both SVG and PNG format.

## Software Tools
I will be using Tensorflow as the primary library for this project. However, the XNOR convolutions are not available in Tensorflow Ops, which means part of this project would be create new Ops in the Tensorflow library.

## Idea


## Milestone
The following tasks should be completed by the milestone on 02/24/2017.
- [ ] XNOR Convolution Ops in Tensorflow
- [ ] A simple 2 layer network on MNIST dataset to test XNOR Convolutions Ops
- [ ] Implement the Sketch-A-Net architecture in Tensorflow

## References
- [Sketch-A-Net]: "Sketch-a-Net that Beats Humans"
- [XNOR-Net]: ImageNet Classification Using Binary Convolutional Neural Networks"
- [Tensorflow]: "TensorFlow: A system for large-scale machine learning"
- [TU-Berlin Sketch Dataset]: "How Do Humans Sketch Objects?"

[Sketch-A-Net]: https://arxiv.org/abs/1501.07873 "Sketch-a-Net that Beats Humans"
[XNOR-Net]: https://arxiv.org/abs/1603.05279 "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks"
[Tensorflow]: https://arxiv.org/abs/1605.08695 "TensorFlow: A system for large-scale machine learning"
[TU-Berlin Sketch Dataset]: http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/ "How Do Humans Sketch Objects?"
