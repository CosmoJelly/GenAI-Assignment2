import tensorflow as tf
import numpy as np
from glob import glob
import os
from PIL import Image

AUTOTUNE = tf.data.AUTOTUNE
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 1

def load_image_pair(sketch_path, photo_path):
    sketch = Image.open(sketch_path).convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
    photo = Image.open(photo_path).convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
    sketch = np.array(sketch) / 127.5 - 1.0
    photo = np.array(photo) / 127.5 - 1.0
    return sketch.astype(np.float32), photo.astype(np.float32)

def load_dataset(split='train'):
    sketch_dir = f"data/{split}/sketches"
    photo_dir = f"data/{split}/photos"
    sketch_paths = sorted(glob(os.path.join(sketch_dir, "*.jpg")))
    photo_paths = sorted(glob(os.path.join(photo_dir, "*.jpg")))

    dataset = tf.data.Dataset.from_tensor_slices((sketch_paths, photo_paths))

    def _parse(s, p):
        sketch, photo = tf.py_function(
            func=lambda s, p: load_image_pair(s.numpy().decode(), p.numpy().decode()),
            inp=[s, p],
            Tout=[tf.float32, tf.float32]
        )
        sketch.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
        photo.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
        return sketch, photo

    dataset = dataset.map(_parse, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).shuffle(100)
    return dataset.prefetch(AUTOTUNE)