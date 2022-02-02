# Deep_SVDD_TF2
Unofficial implementation of 2018 ICML paper "Deep One-Class Classification" 
TensorFlow 2.0 version

Dataset: MNIST

Normal class: 8 (For example)

Steps: train autoencoder first, and then train Deep SVDD one class or soft boundary

May suffer from Hypersphere Collapse

If we set batchnorm trainable == True, the performance will be a little bit better

Github References:

1. https://github.com/lukasruff/Deep-SVDD-PyTorch
2. https://github.com/nuclearboy95/Anomaly-Detection-Deep-SVDD-Tensorflow
