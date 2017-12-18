# tf_depth

This repository implements networks that predict dense disparity maps given the mono/stereo images, based on TensorFlow.

This idea is that to train the network in a self-supervised way. That is, the disparity output by the network is supervised the reconstruction of left image from right in a stereo pair and vise versa. In this framework, the real supervision (i.e. ground-truth disparity) can also be integrated. The disparity can then be used to generate depth map.

* References:
* - Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue
* - Unsupervised Monocular Depth Estimation with Left-Right Consistency
* - Semi-Supervised Deep Learning for Monocular Depth Map Prediction
