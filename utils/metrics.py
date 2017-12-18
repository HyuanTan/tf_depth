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


def depth_metrics(gt_disp, pred_disp, focal, base, base_disp=None, min_depth=1e-3, max_depth=80.0):
    gt_shape = gt_disp.shape
    height = gt_shape[0]
    width = gt_shape[1]
    pred_disp = width * cv2.resize(pred_disp, (width, height), interpolation=cv2.INTER_LINEAR)
    mask_gt = gt_disp > 0
    mask_pred = pred_disp > 0
    mask_pred_eps = np.full(pred_disp.shape, 1e-6)
    mask_pred_eps[mask_pred] = 0
    if base_disp is None:
        mask = np.logical_and(mask_gt, mask_pred)
        rate = 100.0 * mask.sum() / mask_gt.sum()

        gt_depth = focal * base / (gt_disp + (1.0 - mask_gt))
        pred_depth = focal * base / (pred_disp + mask_pred_eps)

        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all_inter = 100.0 * bad_pixels.sum() / mask.sum()

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        thresh = np.maximum((gt_depth[mask] / pred_depth[mask]), (pred_depth[mask] / gt_depth[mask]))
        a1_inter = (thresh < 1.25   ).mean()
        a2_inter = (thresh < 1.25 ** 2).mean()
        a3_inter = (thresh < 1.25 ** 3).mean()

        rmse_inter = (gt_depth[mask] - pred_depth[mask]) ** 2
        rmse_inter = np.sqrt(rmse_inter.mean())

        rmse_log_inter = (np.log(gt_depth[mask]) - np.log(pred_depth[mask])) ** 2
        rmse_log_inter = np.sqrt(rmse_log_inter.mean())

        abs_rel_inter = np.mean(np.abs(gt_depth[mask] - pred_depth[mask]) / gt_depth[mask])

        sq_rel_inter = np.mean(((gt_depth[mask] - pred_depth[mask])**2) / gt_depth[mask])

        mask = gt_disp > 0

        disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all = 100.0 * bad_pixels.sum() / mask.sum()

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


        return rate, d1_all_inter, abs_rel_inter, sq_rel_inter, rmse_inter, rmse_log_inter, a1_inter, a2_inter, a3_inter, d1_all, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    else:
        mask_base = base_disp > 0
        mask_base_eps = np.full(base_disp.shape, 1e-6)
        mask_base_eps[mask_base] = 0

        mask = np.logical_and(mask_gt, np.logical_and(mask_pred, mask_base))
        rate = 100.0 * mask.sum() / mask_gt.sum()

        gt_depth = focal * base / (gt_disp + (1.0 - mask_gt))
        pred_depth = focal * base / (pred_disp + mask_pred_eps)
        base_depth = focal * base / (base_disp + mask_base_eps)

        disp_diff_pred = np.abs(gt_disp[mask] - pred_disp[mask])
        bad_pixels_pred = np.logical_and(disp_diff_pred >= 3, (disp_diff_pred / gt_disp[mask]) >= 0.05)
        d1_all_pred = 100.0 * bad_pixels_pred.sum() / mask.sum()

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        thresh_pred = np.maximum((gt_depth[mask] / pred_depth[mask]), (pred_depth[mask] / gt_depth[mask]))
        a1_pred = (thresh_pred < 1.25   ).mean()
        a2_pred = (thresh_pred < 1.25 ** 2).mean()
        a3_pred = (thresh_pred < 1.25 ** 3).mean()

        rmse_pred = (gt_depth[mask] - pred_depth[mask]) ** 2
        rmse_pred = np.sqrt(rmse_pred.mean())

        rmse_log_pred = (np.log(gt_depth[mask]) - np.log(pred_depth[mask])) ** 2
        rmse_log_pred = np.sqrt(rmse_log_pred.mean())

        abs_rel_pred = np.mean(np.abs(gt_depth[mask] - pred_depth[mask]) / gt_depth[mask])

        sq_rel_pred = np.mean(((gt_depth[mask] - pred_depth[mask])**2) / gt_depth[mask])

        disp_diff_base = np.abs(gt_disp[mask] - base_disp[mask])
        bad_pixels_base = np.logical_and(disp_diff_base >= 3, (disp_diff_base / gt_disp[mask]) >= 0.05)
        d1_all_base = 100.0 * bad_pixels_base.sum() / mask.sum()

        base_depth[base_depth < min_depth] = min_depth
        base_depth[base_depth > max_depth] = max_depth

        thresh_base = np.maximum((gt_depth[mask] / base_depth[mask]), (base_depth[mask] / gt_depth[mask]))
        a1_base = (thresh_base < 1.25   ).mean()
        a2_base = (thresh_base < 1.25 ** 2).mean()
        a3_base = (thresh_base < 1.25 ** 3).mean()

        rmse_base = (gt_depth[mask] - base_depth[mask]) ** 2
        rmse_base = np.sqrt(rmse_base.mean())

        rmse_log_base = (np.log(gt_depth[mask]) - np.log(base_depth[mask])) ** 2
        rmse_log_base = np.sqrt(rmse_log_base.mean())

        abs_rel_base = np.mean(np.abs(gt_depth[mask] - base_depth[mask]) / gt_depth[mask])

        sq_rel_base = np.mean(((gt_depth[mask] - base_depth[mask])**2) / gt_depth[mask])

        return rate, d1_all_pred, abs_rel_pred, sq_rel_pred, rmse_pred, rmse_log_pred, a1_pred, a2_pred, a3_pred, d1_all_base, abs_rel_base, sq_rel_base, rmse_base, rmse_log_base, a1_base, a2_base, a3_base
