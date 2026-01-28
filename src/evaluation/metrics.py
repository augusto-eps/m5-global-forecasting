from typing import Union
import pandas as pd
import numpy as np

def wape(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    eps: float = 1e-8,
    mask_zeros: bool = True
) -> float:
    """
    Weighted Absolute Percentage Error (WAPE) with guardrails for zeros.

    WAPE = sum(|y_true - y_pred|) / sum(|y_true|)

    If mask_zeros=True, time steps where y_true == 0 are excluded
    from both numerator and denominator.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    eps : float, optional
        Small constant to avoid division by zero.
    mask_zeros : bool, optional
        Whether to ignore zero target values.

    Returns
    -------
    float
        WAPE value. Returns np.nan if denominator is zero.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if mask_zeros:
        mask = y_true != 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))

    if denominator < eps:
        return np.nan
    
    else:
        return numerator / denominator

