import tensorflow as tf
import os

def get_signature_dataset(data_dir, img_size=64, batch_size=64, train_split=0.9):
    # Load all image paths
    all_images = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                all_images.append(os.path.join(root, f))

    total = len(all_images)
    split = int(total * train_split)
    train_paths = all_images[:split]
    test_paths = all_images[split:]

    def preprocess(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = (img - 127.5) / 127.5
        return img

    # Augmentations for training
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.image.random_jpeg_quality(img, 90, 100)
        img = tf.image.random_crop(tf.image.resize_with_crop_or_pad(img, img_size+4, img_size+4), [img_size, img_size, 1])
        return img

    train_ds = (tf.data.Dataset.from_tensor_slices(train_paths)
                .shuffle(1000)
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    test_ds = (tf.data.Dataset.from_tensor_slices(test_paths)
               .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))
    return train_ds, test_ds