# -*- coding: utf-8 -*-
"""
Functions for synthetic population metrics
"""
import numpy as np
import math
import pandas as pd
import scipy.stats as ss
from collections import Counter


def total_absolute_error(df_o, df_s):
    """
    Parameters:
    df_o ...... original data
    df_s ...... synthetic data
    Returns:
    total_absolute_error ..... error for each feature in absolute counts
    """
    return abs((df_o.values - df_s.values).sum())


def relative_absolute_error(df_o, df_s):
    return abs((df_o.values - df_s.values).sum() * 1/df_o.shape[1])


def ci_intervals_counts(df_o, df_s, limit=1.96):
    real_mean = df_o.values.mean()
    real_std = df_o.values.std()
    ebm = limit * real_std/np.sqrt(df_o.shape[1])
    upper_ci = real_mean + ebm
    lower_ci = real_mean - ebm
    # Count how many variables are inside and outside
    s_mean = df_s.mean()
    s_under = (s_mean < lower_ci).sum()
    s_over = (s_mean > upper_ci).sum()
    return lower_ci, upper_ci, s_under, s_over


def z_score(df_o, df_s):
    synthetic_mean = df_s.values.mean()
    synthetic_std = df_s.values.std()
    z_synthetic = (df_s.values - synthetic_mean) / synthetic_std
    original_mean = df_o.values.mean()
    original_std = df_o.values.std()
    z_original = (df_o.values - original_mean) / original_std
    return (z_original, original_std), (z_synthetic,  synthetic_std)


def cramers_v(df_o_cat, df_s_cat):
    """
    Calculate Cramers V from two categorised pandas dataframe columns.
    Parameters:
    df_o_cat ...... one column from first dataset
    df_s_cat ...... one (different) column from first dataset
                    or "same" column from different dataset
    """
    confusion_matrix = pd.crosstab(df_o_cat, df_s_cat).to_numpy()
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, c = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((c - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    ccorr = c - ((c - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((ccorr - 1), (rcorr - 1)))


def theils_u(x, y):
    s_xy = conditional_entropy(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def conditional_entropy(df_o, df_s):
    log_base = 10.0  # or 2.0 or e
    y_counter = Counter(df_s)
    xy_counter = Counter(list(zip(df_o, df_s)))
    total_occurrences = sum(y_counter.values())
    entropy = 0.0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y / p_xy, log_base)
    return entropy


def tetrachoric(df_o_binary, df_s_binary):
    a = df_o_binary.sum()
    d = df_s_binary.sum()
    n_o = df_o_binary.shape[0]
    n_s = df_s_binary.shape[0]
    t_o = np.cos((np.pi / 1) * np.sqrt(a * d / (n_s - d) / (n_o - a)))
    return t_o 


def accuracy(real, replica, N, upscale=1):
    """
    real ..... dataframe with sliced original data
    replica .. dataframe with sliced synthetic data
    N ........ total number of examples in original data
    """
    tp = real.shape[0]
    tn = N - tp
    diff = real.shape[0] - replica.shape[0] / upscale
    if diff < 0:
        fp = abs(diff)
        fn = 0
    else:
        fn = diff
        fp = 0
    return (tp + tn) / (tp + fn + tn + fp)


def precision(real, replica, N, upscale=1):
    """
    real ..... dataframe with sliced original data
    replica .. dataframe with sliced synthetic data
    N ........ total number of examples in original data
    """
    tp = real.shape[0]
    tn = N - tp
    diff = real.shape[0] - replica.shape[0] / upscale
    if diff < 0:
        fp = abs(diff)
        fn = 0
    else:
        fn = diff
        fp = 0
    return tp / (fp + tp)


def recall(real, replica, N, upscale=1):
    """
    real ..... dataframe with sliced original data
    replica .. dataframe with sliced synthetic data
    N ........ total number of examples in original data
    """
    tp = real.shape[0]
    tn = N - tp
    diff = real.shape[0] - replica.shape[0] / upscale
    if diff < 0:
        fp = abs(diff)
        fn = 0
    else:
        fn = diff
        fp = 0
    return tp / (fn + tp)


def F1(real, replica, N, upscale=1):
    """
    F1 balance the precision and recall as these are inverse related:

    real ..... dataframe with sliced original data
    replica .. dataframe with sliced synthetic data
    N ........ total number of examples in original data
    """
    precision_score = precision(real, replica, N, upscale)
    recall_score = recall(real, replica, N, upscale)
    return 2 * precision_score * recall_score / (precision_score + recall_score)


def misclassification(real, replica, N, upscale=1):
    """
    real ..... dataframe with sliced original data
    replica .. dataframe with sliced synthetic data
    N ........ total number of examples in original data
    """
    tp = real.shape[0]
    tn = N - tp
    diff = real.shape[0] - replica.shape[0] / upscale
    if diff < 0:
        fp = abs(diff)
        fn = 0
    else:
        fn = diff
        fp = 0
    return (fp + fn) / (tp + fn + tn + fp)


def get_bivariate(df1, df2):
    m1 = (df1.sum().values / df1.shape[0]).reshape(-1, 1)
    m2 = (df2.sum().values / df2.shape[0]).reshape(-1, 1)
    cross1 = (m1 @ m1.T)
    cross2 = (m1 @ m2.T)
    r1 = pd.DataFrame(cross1,
                      columns=df1.columns,
                      index=df1.columns)
    r2 = pd.DataFrame(cross2,
                      columns=df1.columns,
                      index=df1.columns)
    r3 = pd.DataFrame(m1,
                      columns=["Original"],
                      index=df1.columns)
    r4 = pd.DataFrame(m2,
                      columns=["Synthetic"],
                      index=df1.columns)
    return r1, r2, r3, r4
