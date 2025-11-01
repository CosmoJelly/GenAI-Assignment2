import tensorflow as tf
from utils.dataset import get_signature_dataset

if __name__ == "__main__":
    data_dir = "sign_data/train"  # adjust if needed
    train_ds, test_ds = get_signature_dataset(data_dir)

    print("✅ Dataset loaded successfully!")
    print(f"Train batches: {len(list(train_ds))}")
    print(f"Test batches:  {len(list(test_ds))}")

    for images in train_ds.take(1):
        print(f"Batch shape: {images.shape}")
        print(f"Image dtype: {images.dtype}")
        print(f"Value range: min={tf.reduce_min(images).numpy():.3f}, max={tf.reduce_max(images).numpy():.3f}")
        break