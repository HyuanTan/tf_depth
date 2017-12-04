#!/usr/bin/python
import cv2
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


def depth_metrics(gt_disp, pred_disp, focal, base, min_depth=1e-3, max_depth=80.0):
    gt_shape = gt_disp.shape
    height = gt_shape[0]
    width = gt_shape[1]
    pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = gt_disp > 0
    mask_pred = pred_disp > 0

    disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
    bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
    d1_all = 100.0 * bad_pixels.sum() / mask.sum()

    gt_depth = focal * base / (gt_disp + (1.0 - mask))
    pred_depth = focal * base / pred_disp

    pred_depth[pred_depth < min_depth] = min_depth
    pred_depth[pred_depth > max_depth] = max_depth

    thresh = np.maximum((gt_depth[mask] / pred_depth[mask]), (pred_depth[mask] / gt_depth[mask]))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt_depth[mask] - pred_depth[mask]) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt_depth[mask]) - np.log(pred_depth[mask])) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt_depth[mask] - pred_depth[mask]) / gt_depth[mask])

    sq_rel = np.mean(((gt_depth[mask] - pred_depth[mask])**2) / gt_depth[mask])

    return d1_all, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

