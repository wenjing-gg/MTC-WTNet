import numpy as np
from scipy import ndimage as ndi

def assd(pred, gt):
    """
    Calculate Average Symmetric Surface Distance (ASSD).
    Applicable to 2D or 3D binary masks, with added input and output validity checks.

    Parameters:
    -----------
    pred : np.ndarray
        Predicted binary mask with values {0,1}, can be 2D or 3D.
    gt   : np.ndarray
        Ground truth binary mask with values {0,1}, can be 2D or 3D.

    Returns:
    --------
    asd : float
        Average Symmetric Surface Distance (ASSD), returns 0.0 if exception occurs
    """
    # =============== 1) Input validity check ===============
    # 1.1) Check if shapes are consistent
    if pred.shape != gt.shape:
        raise ValueError(f"Predicted mask shape {pred.shape} does not match ground truth mask shape {gt.shape}.")

    # 1.2) Check if dimension is 2D or 3D
    if pred.ndim not in (2, 3):
        raise ValueError(f"Only 2D or 3D data is supported, but current input dimension is {pred.ndim}D.")

    # 1.3) Convert to bool type and check if only contains 0, 1
    pred_unique = np.unique(pred)
    gt_unique = np.unique(gt)
    if not set(pred_unique).issubset({0, 1}) or not set(gt_unique).issubset({0, 1}):
        raise ValueError("Input masks should only contain binary values {0, 1}.")
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    # =============== 2) Special case handling ===============
    # If both are empty (all 0), distance is 0
    if not pred.any() and not gt.any():
        return 0.0

    # =============== 3) Calculate distance ===============
    try:
        # 3.1) Extract respective boundary points
        pred_surface = _extract_surface(pred)
        gt_surface   = _extract_surface(gt)

        # 3.2) Calculate distance transform
        dist_pred = ndi.distance_transform_edt(~pred)
        dist_gt   = ndi.distance_transform_edt(~gt)

        # 3.3) Calculate distance from pred boundary to gt, and gt boundary to pred
        dists_pred_to_gt = dist_gt[pred_surface]
        dists_gt_to_pred = dist_pred[gt_surface]

        # 3.4) Calculate average separately, then take average of both
        asd_pred_to_gt = dists_pred_to_gt.mean()  # Average distance (predicted boundary to GT)
        asd_gt_to_pred = dists_gt_to_pred.mean()  # Average distance (GT boundary to predicted)
        asd = (asd_pred_to_gt + asd_gt_to_pred) / 2.0

    except Exception as e:
        print(f"[Warning] ASSD calculation exception: {e}")
        return 0.0  # or return np.nan, depending on requirements

    # =============== 4) Output check and return ===============
    # If calculation result has NaN or negative value (theoretically shouldn't happen, but just in case)
    if np.isnan(asd) or asd < 0:
        print("[Warning] ASSD calculation result has NaN or negative value, automatically set to 0.0")
        asd = 0.0

    return float(asd)


def hd95(pred, gt):
    """
    Calculate 95% Hausdorff Distance (HD95).
    Applicable to 2D or 3D binary masks, with added input and output validity checks.

    Parameters:
    -----------
    pred : np.ndarray
        Predicted binary mask with values {0,1}, can be 2D or 3D.
    gt   : np.ndarray
        Ground truth binary mask with values {0,1}, can be 2D or 3D.

    Returns:
    --------
    hd_95 : float
        95% Hausdorff distance, returns 0.0 if exception occurs
    """
    # =============== 1) Input validity check ===============
    if pred.shape != gt.shape:
        raise ValueError(f"Predicted mask shape {pred.shape} does not match ground truth mask shape {gt.shape}.")
    if pred.ndim not in (2, 3):
        raise ValueError(f"Only 2D or 3D data is supported, but current input dimension is {pred.ndim}D.")

    pred_unique = np.unique(pred)
    gt_unique = np.unique(gt)
    if not set(pred_unique).issubset({0, 1}) or not set(gt_unique).issubset({0, 1}):
        raise ValueError("Input masks should only contain binary values {0, 1}.")
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    if not pred.any() and not gt.any():
        return 0.0

    # =============== 2) Calculate HD95 ===============
    try:
        pred_surface = _extract_surface(pred)
        gt_surface   = _extract_surface(gt)

        dist_pred = ndi.distance_transform_edt(~pred)
        dist_gt   = ndi.distance_transform_edt(~gt)

        dists_pred_to_gt = dist_gt[pred_surface]
        dists_gt_to_pred = dist_pred[gt_surface]

        # Merge distances, take 95% percentile
        all_dist = np.concatenate([dists_pred_to_gt, dists_gt_to_pred], axis=None)
        hd_95 = np.percentile(all_dist, 95)
    except Exception as e:
        print(f"[Warning] HD95 calculation exception: {e}")
        return 0.0  # or return np.nan, depending on requirements

    # =============== 3) Output check and return ===============
    if np.isnan(hd_95) or hd_95 < 0:
        print("[Warning] HD95 calculation result has NaN or negative value, automatically set to 0.0")
        hd_95 = 0.0

    return float(hd_95)


def _extract_surface(mask):
    """
    Extract boundary points (surface) of binary mask.
    mask is a bool type 2D/3D array.

    Approach: XOR mask with its eroded result to get boundary positions.
    Final return value is a tuple (idx1, idx2, ...) that can be used to index into mask.
    """
    eroded_mask = ndi.binary_erosion(mask)
    surface = np.logical_xor(mask, eroded_mask)
    return np.where(surface)

