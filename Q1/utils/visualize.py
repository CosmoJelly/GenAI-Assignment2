import matplotlib.pyplot as plt
import numpy as np

def show_images(images, n=8):
    plt.figure(figsize=(10, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        img = images[i].numpy() if hasattr(images[i], "numpy") else np.array(images[i])
        plt.imshow(np.squeeze(img), cmap='gray')
        plt.axis('off')
    plt.show()