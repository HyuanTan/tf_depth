#!/usr/bin/python
import numpy as np

def _fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def get_class_hist(predictions, labels):
    size = predictions.shape[0]
    num_class = predictions.shape[3]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
        hist += _fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)

    return hist

def per_class_iu(predictions, labels):
    hist = get_class_hist(predictions, labels)

    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    return mean_iu, iu

def per_class_iu_from_hist(class_hist):
    iu = np.diag(class_hist) / (class_hist.sum(1) + class_hist.sum(0) - np.diag(class_hist))
    mean_iu = np.nanmean(iu)
    return mean_iu, iu


