import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_ssim(imgs_a, imgs_b):
    imgs_a = ((imgs_a + 1.0) / 2.0).numpy()
    imgs_b = ((imgs_b + 1.0) / 2.0).numpy()
    scores = []
    for i in range(len(imgs_a)):
        s = ssim(imgs_a[i].squeeze(), imgs_b[i].squeeze(), data_range=1.0)
        scores.append(s)
    return np.mean(scores)