import os
import pickle
import numpy as np
import tensorflow as tf

IMG_SHAPE = (32, 32, 3)
BATCH_SIZE = 64

def load_cifar10_batch(batch_filename):
    """Loads a single CIFAR-10 batch file."""
    with open(batch_filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels

def load_cifar10_cats_dogs(data_dir="cifar-10-batches-py"):
    """Loads only the cat (3) and dog (5) classes from local CIFAR-10 files."""
    images, labels = [], []

    # Load all training batches
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        batch_images, batch_labels = load_cifar10_batch(batch_path)
        images.append(batch_images)
        labels.extend(batch_labels)

    images = np.concatenate(images)
    labels = np.array(labels)

    # Filter for cats (3) and dogs (5)
    mask = np.isin(labels, [3, 5])
    images = images[mask]
    labels = labels[mask]

    # Normalize to [-1, 1] for use with tanh output generator
    images = images.astype("float32") / 127.5 - 1.0

    # Build TensorFlow dataset pipeline
    dataset = (
        tf.data.Dataset.from_tensor_slices(images)
        .shuffle(buffer_size=1000)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"Loaded {len(images)} cat/dog images from local CIFAR-10 directory.")
    return dataset  # type: tf.data.Dataset

if __name__ == "__main__":
    ds = load_cifar10_cats_dogs()
    for batch in ds.take(1):
        print("Batch shape:", batch.shape, "Pixel range:", (batch.numpy().min(), batch.numpy().max()))