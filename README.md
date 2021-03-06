# tf_depth

This repository implements networks that predict dense disparity maps given the mono/stereo images, based on TensorFlow.

This idea is that to train the network in a self-supervised way. That is, the disparity output by the network is supervised the reconstruction of left image from right in a stereo pair and vise versa. In this framework, the real supervision (i.e. ground-truth disparity) can also be integrated. The disparity can then be used to generate depth map.

*References:*
* *Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue*
* *Unsupervised Monocular Depth Estimation with Left-Right Consistency*
* *Semi-Supervised Deep Learning for Monocular Depth Map Prediction*

### Some metrics

The following metrics is evaluated on Kitti stereo 2015 dataset which has good ground truth.

The baseline is OpenCV stereoBM algorithm. Our Mono/Stereo networks are trained in total self-supervised way.

| | OpenCV stereoBM | Mono Network | Stereo Network |
|---|---|---|---|
|detection_rate| 36.806% | 100% | 100% |
|RMSE| 4.983 | 5.551 | 4.409 |
|abs_relative_err| 0.075 | 0.110 | 0.067 |
|sqr_relative_err| 1.261 | 1.113 | 0.926 |
|RMSE_total| 53.371 | 5.551 | 4.409 |
|abs_relative_err_total| 4.335 | 0.110 | 0.067 |
|sqr_relative_err_total| 303.102 | 1.113 | 0.926 |


The detection_rate is ratio that pixels having disparities computed by the algorithms compared to ground truth.
Thus, metrics with "_total" suffix are computed on all ground-truth pixels, metrics without "_total" suffix are computed on pixels having algo-detected disparities.

Since our networks can output dense disparity maps, the detection rate is 100%, much higher than the baseline, and also the metrics on all ground-truth pixels are much better than the baseline.

Better results can be obtained by integrating ground-truth disparity in model training.

