# Neural Style Transfer
An implementation of the arXiv preprint [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

## The Problem
Given a _content image **C**_ and a _style image **S**_, the style transfer problem consists in finding a _target image **T**_, that has the same content of the image **C** and the same style of the image **S**.
Although this problem is relatively simple to formulate, it was not mathematically defined before 2015.
What does it mean that two images have the same content or style? What is the content of an image and what is the style? 
Gatys et all. were able to answer these questions in the paper “A neural Algorithm of Artistic Style”. 

![StyleTransfer](https://www.dropbox.com/s/lj1p1o61x69424p/styletransferexample.png?raw=1)
