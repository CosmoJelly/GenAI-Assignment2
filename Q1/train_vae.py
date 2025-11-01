import tensorflow as tf
import os
from models.vae import VAE
from utils.dataset import get_signature_dataset
from utils.visualize import show_images
from utils.metrics import compute_ssim
import config

os.makedirs(config.OUTPUT_DIR, exist_ok=True)

train_ds, test_ds = get_signature_dataset(config.DATA_DIR, config.IMG_SIZE, config.BATCH_SIZE)
vae = VAE(config.Z_DIM)
optimizer = tf.keras.optimizers.Adam(config.VAE_LR)

for epoch in range(config.EPOCHS_VAE):
    for batch in train_ds:
        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss = vae.compute_loss(batch)
        grads = tape.gradient(total_loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    print(f"Epoch {epoch+1}/{config.EPOCHS_VAE}: Total={total_loss:.4f}, Recon={recon_loss:.4f}, KL={kl_loss:.4f}")

    # Show sample reconstructions
    for test_batch in test_ds.take(1):
        recon, _, _ = vae(test_batch)
        #show_images(recon[:8])
        break

# Evaluate SSIM
for test_batch in test_ds.take(1):
    recon, _, _ = vae(test_batch)
    ssim_score = compute_ssim(test_batch, recon)
    print("VAE Test SSIM:", ssim_score)

show_images(recon[:8])