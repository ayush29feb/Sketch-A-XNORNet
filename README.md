# Sketch-A-XNORNet
An implementation of a variation of [Sketch-A-Net] using [XNOR-Net] in [TensorFlow].

**Sketch-A-Net** is a multi-scale multi-channel deep neural network framework that, for the first time, yields sketch recognition performance surpassing that of humans.

**XNOR-Net** is a variation of standard convolutional neural networks with an approximates convolutions using primarily binary operations. XNOR-Net approximates the weights and the input tensors to be binary numbers which allows faster computation allowing 32x memory saving and 58x faster convolution operations.

## Idea
The core idea of this project is to create a free-hand drawn sketch classification algorithm that understands the object drawn in the sketch by a human being. The core motivation for this idea is relevant is the context of Human Computer Interfaces. As we are moving towards more natural forms of computing interfaces like AR/VR that act as great output interfaces, research in better input sources is still lacking other than voice and natural language recognition systems. 

Sketching is one of the most natural form of communication among humans since historic times. With an effective yet efficient sketch classification system we would be able to initiate a whole new form of input interface for computing platforms like AR headsets (Microsoft Hololens) or ever Mobile AR like (Google Tango). In this project we aim to extend the research done in [Sketch-A-Net] and [XNOR-Net] to create a efficient sketch classification algorithm.

## Data
I will be using the [TU-Berlin Sketch Dataset], which is the most commonly used human sketch dataset. It contains 250 categories with 80 sketches per category. It was collected on Amazon Mechanical Turk from 1350 participants, thus providing a variety of sketches for each category. The images are available in both SVG and PNG format.

## Software Tools
I will be using [Tensorflow] as the primary library for this project. However, the XNOR convolutions are not yet available in Tensorflow Ops, which means part of this project would be create new Ops in the Tensorflow library.

## Milestone
The following tasks should be completed by the milestone on 02/24/2017.
- [ ] XNOR Convolution Ops in Tensorflow
- [x] Implement the Sketch-A-Net architecture in Tensorflow

## References
- [Sketch-A-Net]: "Sketch-a-Net that Beats Humans"
- [XNOR-Net]: ImageNet Classification Using Binary Convolutional Neural Networks"
- [Tensorflow]: "TensorFlow: A system for large-scale machine learning"
- [TU-Berlin Sketch Dataset]: "How Do Humans Sketch Objects?"

[Sketch-A-Net]: https://arxiv.org/abs/1501.07873 "Sketch-a-Net that Beats Humans"
[XNOR-Net]: https://arxiv.org/abs/1603.05279 "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks"
[Tensorflow]: https://arxiv.org/abs/1605.08695 "TensorFlow: A system for large-scale machine learning"
[TU-Berlin Sketch Dataset]: http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/ "How Do Humans Sketch Objects?"
