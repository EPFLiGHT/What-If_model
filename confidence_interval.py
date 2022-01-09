import numpy as np

def confidence_interval(values):
    x = np.nanmean(values)
    std = np.nanstd(values)
    
    return x.round(3), (x - 1.96 * std / np.sqrt(len(values))).round(3), (x + 1.96 * std / np.sqrt(len(values))).round(3)