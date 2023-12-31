# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

__all__ = [
    'load_heartrate'
]


def load_heartrate(as_series=False):
    """MIT heart-rate data.

    A sample of heartrate data borrowed from an
    `MIT database <http://ecg.mit.edu/time-series/>`_. The sample consists
    of 150 evenly spaced (0.5 seconds) heartrate measurements.

    Parameters
    ----------
    as_series : bool, optional (default=False)
        Whether to return a Pandas series. If False, will return a 1d
        numpy array.

    Returns
    -------
    rslt : array-like, shape=(n_samples,)
        The heartrate vector.

    Examples
    --------
    >>> from Clustering.datasets import load_heartrate
    >>> load_heartrate()
    array([84.2697, 84.2697, 84.0619, 85.6542, 87.2093, 87.1246, 86.8726,
           86.7052, 87.5899, 89.1475, 89.8204, 89.8204, 90.4375, 91.7605,
           93.1081, 94.3291, 95.8003, 97.5119, 98.7457, 98.904 , 98.3437,
           98.3075, 98.8313, 99.0789, 98.8157, 98.2998, 97.7311, 97.6471,
           97.7922, 97.2974, 96.2042, 95.2318, 94.9367, 95.0867, 95.389 ,
           95.5414, 95.2439, 94.9415, 95.3557, 96.3423, 97.1563, 97.4026,
           96.7028, 96.5516, 97.9837, 98.9879, 97.6312, 95.4064, 93.8603,
           93.0552, 94.6012, 95.8476, 95.7692, 95.9236, 95.7692, 95.9211,
           95.8501, 94.6703, 93.0993, 91.972 , 91.7821, 91.7911, 90.807 ,
           89.3196, 88.1511, 88.7762, 90.2265, 90.8066, 91.2284, 92.4238,
           93.243 , 92.8472, 92.5926, 91.7778, 91.2974, 91.6364, 91.2952,
           91.771 , 93.2285, 93.3199, 91.8799, 91.2239, 92.4055, 93.8716,
           94.5825, 94.5594, 94.9453, 96.2412, 96.6879, 95.8295, 94.7819,
           93.4731, 92.7997, 92.963 , 92.6996, 91.9648, 91.2417, 91.9312,
           93.9548, 95.3044, 95.2511, 94.5358, 93.8093, 93.2287, 92.2065,
           92.1588, 93.6376, 94.899 , 95.1592, 95.2415, 95.5414, 95.0971,
           94.528 , 95.5887, 96.4715, 96.6158, 97.0769, 96.8531, 96.3947,
           97.4291, 98.1767, 97.0148, 96.044 , 95.9581, 96.4814, 96.5211,
           95.3629, 93.5741, 92.077 , 90.4094, 90.1751, 91.3312, 91.2883,
           89.0592, 87.052 , 86.6226, 85.7889, 85.6348, 85.3911, 83.8064,
           82.8729, 82.6266, 82.645 , 82.645 , 82.645 , 82.645 , 82.645 ,
           82.645 , 82.645 , 82.645 ])

    >>> load_heartrate(True).head()
    0    84.2697
    1    84.2697
    2    84.0619
    3    85.6542
    4    87.2093

    References
    ----------
    .. [1] Goldberger AL, Rigney DR. Nonlinear dynamics at the bedside.
           In: Glass L, Hunter P, McCulloch A, eds.
           Theory of Heart: Biomechanics, Biophysics, and Nonlinear
           Dynamics of Cardiac Function. New York: Springer-Verlag, 1991,
           pp. 583-605.
    """
    rslt = np.array([
        84.2697, 84.2697, 84.0619, 85.6542, 87.2093, 87.1246,
        86.8726, 86.7052, 87.5899, 89.1475, 89.8204, 89.8204,
        90.4375, 91.7605, 93.1081, 94.3291, 95.8003, 97.5119,
        98.7457, 98.904, 98.3437, 98.3075, 98.8313, 99.0789,
        98.8157, 98.2998, 97.7311, 97.6471, 97.7922, 97.2974,
        96.2042, 95.2318, 94.9367, 95.0867, 95.389, 95.5414,
        95.2439, 94.9415, 95.3557, 96.3423, 97.1563, 97.4026,
        96.7028, 96.5516, 97.9837, 98.9879, 97.6312, 95.4064,
        93.8603, 93.0552, 94.6012, 95.8476, 95.7692, 95.9236,
        95.7692, 95.9211, 95.8501, 94.6703, 93.0993, 91.972,
        91.7821, 91.7911, 90.807, 89.3196, 88.1511, 88.7762,
        90.2265, 90.8066, 91.2284, 92.4238, 93.243, 92.8472,
        92.5926, 91.7778, 91.2974, 91.6364, 91.2952, 91.771,
        93.2285, 93.3199, 91.8799, 91.2239, 92.4055, 93.8716,
        94.5825, 94.5594, 94.9453, 96.2412, 96.6879, 95.8295,
        94.7819, 93.4731, 92.7997, 92.963, 92.6996, 91.9648,
        91.2417, 91.9312, 93.9548, 95.3044, 95.2511, 94.5358,
        93.8093, 93.2287, 92.2065, 92.1588, 93.6376, 94.899,
        95.1592, 95.2415, 95.5414, 95.0971, 94.528, 95.5887,
        96.4715, 96.6158, 97.0769, 96.8531, 96.3947, 97.4291,
        98.1767, 97.0148, 96.044, 95.9581, 96.4814, 96.5211,
        95.3629, 93.5741, 92.077, 90.4094, 90.1751, 91.3312,
        91.2883, 89.0592, 87.052, 86.6226, 85.7889, 85.6348,
        85.3911, 83.8064, 82.8729, 82.6266, 82.645, 82.645,
        82.645, 82.645, 82.645, 82.645, 82.645, 82.645])

    if as_series:
        return pd.Series(rslt)
    return rslt
