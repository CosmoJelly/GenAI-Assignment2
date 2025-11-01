import matplotlib.pyplot as plt
import numpy as np

def display_predictions(generator, dataset, num=3):
    for sketch, photo in dataset.take(1):
        pred = generator(sketch, training=False)
        plt.figure(figsize=(12, 4))
        for i in range(min(num, sketch.shape[0])):
            imgs = [sketch[i], photo[i], pred[i]]
            titles = ["Sketch", "Real", "Generated"]
            for j, (img, title) in enumerate(zip(imgs, titles)):
                plt.subplot(num, 3, i * 3 + j + 1)
                plt.imshow((img + 1) / 2)
                plt.title(title)
                plt.axis('off')
        plt.show()