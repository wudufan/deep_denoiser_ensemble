import numpy as np

def pad_projection_b(prj_a, prj_b, n_pixels = 20, reg = 0.1):
    margin = int((prj_a.shape[2] - prj_b.shape[2]) / 2)
    
    prj_b_left = prj_b[..., :n_pixels]
    prj_a_left = prj_a[..., margin:margin+n_pixels]
    coefs_left = np.sum(prj_b_left * prj_a_left, 2) / np.sum(prj_b_left * prj_a_left + reg, 2)
    
    prj_b_right = prj_b[..., -n_pixels:]
    prj_a_right = prj_a[..., -margin-n_pixels:-margin]
    coefs_right = np.sum(prj_b_right * prj_a_right, 2) / np.sum(prj_b_right * prj_a_right + reg, 2)
    
    extrap_prj_b = np.zeros_like(prj_a)
    extrap_prj_b[..., :margin] = prj_a[..., :margin] * coefs_left[..., np.newaxis]
    extrap_prj_b[..., margin:-margin] = prj_b
    extrap_prj_b[..., -margin:] = prj_a[..., -margin:] * coefs_right[..., np.newaxis]
    
    return extrap_prj_b