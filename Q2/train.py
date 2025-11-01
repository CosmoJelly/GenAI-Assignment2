import tensorflow as tf
from generator import build_generator
from discriminator import build_siamese_discriminator
from data_loader import load_cifar10_cats_dogs
import numpy as np
import os
import matplotlib.pyplot as plt

LATENT_DIM = 100
EPOCHS = 500
SAVE_INTERVAL = 5
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# hyperparams
LABEL_SMOOTH_REAL = 0.9
LABEL_SMOOTH_FAKE = 0.1
LABEL_FLIP_PROB = 0.03
DISCRIMINATOR_STEPS = 1
LAMBDA_FM = 0.08
LAMBDA_MS = 0.08

def save_image_grid(images, epoch, n=5):
    images = tf.clip_by_value((images * 0.5) + 0.5, 0., 1.)
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            axes[i,j].imshow(images[idx])
            axes[i,j].axis('off')
            idx += 1
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/epoch_{epoch+1:03d}.png")
    plt.close(fig)

def mode_seeking_loss(gz1, gz2, z1, z2, eps=1e-8):
    num = tf.reduce_mean(tf.abs(gz1 - gz2), axis=[1,2,3])
    den = tf.reduce_mean(tf.abs(z1 - z2), axis=1) + eps
    ratio = num / den
    return -tf.reduce_mean(ratio)

def train():
    dataset = load_cifar10_cats_dogs()
    generator = build_generator(LATENT_DIM)
    discriminator = build_siamese_discriminator()

    gen_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, clipnorm=1.0)
    disc_opt = tf.keras.optimizers.Adam(1e-5, beta_1=0.5, clipnorm=1.0)

    # try get feature extractor for fm
    feat_model = None
    try:
        feat_model = discriminator.get_layer('feature_extractor')
    except Exception:
        pass

    fixed_noise = tf.random.normal([25, LATENT_DIM])

    @tf.function
    def d_step(real_images, z):
        bs = tf.shape(real_images)[0]
        fake = generator(z, training=True)

        real_noisy = real_images + tf.random.normal(tf.shape(real_images), 0.0, 0.05)
        fake_noisy = fake + tf.random.normal(tf.shape(fake), 0.0, 0.05)

        real_labels = tf.ones((bs,1), dtype=tf.float32) * LABEL_SMOOTH_REAL
        fake_labels = tf.zeros((bs,1), dtype=tf.float32) + LABEL_SMOOTH_FAKE

        # occasional label flip
        if tf.random.uniform(()) < LABEL_FLIP_PROB:
            tmp = real_labels
            real_labels = fake_labels
            fake_labels = tmp

        with tf.GradientTape() as tape:
            sim_rf = discriminator([real_noisy, fake_noisy], training=True)
            sim_rr = discriminator([real_noisy, real_noisy], training=True)

            # LSGAN style: push sim_rf toward fake_labels (low), sim_rr toward real_labels (high)
            loss_rf = tf.reduce_mean(tf.square(sim_rf - fake_labels))
            loss_rr = tf.reduce_mean(tf.square(sim_rr - real_labels))
            d_loss = 0.5 * (loss_rf + loss_rr)

            # regularize to avoid collapse to constant 0.5
            d_reg = 0.05 * tf.reduce_mean(tf.square(sim_rf - 0.5))
            d_loss = d_loss + d_reg

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        disc_opt.apply_gradients(zip(grads, discriminator.trainable_variables))
        return d_loss, tf.reduce_mean(sim_rf), tf.reduce_mean(sim_rr)

    @tf.function
    def g_step(real_images, z, z2):
        bs = tf.shape(real_images)[0]
        with tf.GradientTape() as tape:
            fake = generator(z, training=True)
            sim = discriminator([real_images, fake], training=True)
            g_loss = tf.reduce_mean(tf.square(sim - 1.0))  # push toward 1

            # feature matching
            if feat_model is not None:
                real_feats = feat_model(real_images, training=False)
                fake_feats = feat_model(fake, training=False)
                fm = tf.reduce_mean(tf.abs(real_feats - fake_feats))
                g_loss += LAMBDA_FM * fm

            # mode seeking
            if z2 is not None:
                gz1 = generator(z, training=True)
                gz2 = generator(z2, training=True)
                ms = mode_seeking_loss(gz1, gz2, z, z2)
                g_loss += LAMBDA_MS * ms

        grads = tape.gradient(g_loss, generator.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        gen_opt.apply_gradients(zip(grads, generator.trainable_variables))
        return g_loss, tf.reduce_mean(sim)

    for epoch in range(EPOCHS):
        d_losses, g_losses, sims = [], [], []
        for real_batch in dataset:
            bs = tf.shape(real_batch)[0]
            z = tf.random.normal([bs, LATENT_DIM])
            z2 = tf.random.normal([bs, LATENT_DIM])

            # D updates
            for _ in range(DISCRIMINATOR_STEPS):
                d_loss, sim_rf_avg, sim_rr_avg = d_step(real_batch, z)
            # G update
            g_loss, gen_sim = g_step(real_batch, z, z2)

            d_losses.append(d_loss)
            g_losses.append(g_loss)
            sims.append(gen_sim)

        avg_d = np.mean([float(x) for x in d_losses])
        avg_g = np.mean([float(x) for x in g_losses])
        avg_sim = np.mean([float(x) for x in sims])

        print(f"Epoch {epoch+1:03d}/{EPOCHS} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} | Avg Similarity: {avg_sim:.4f}")

        if (epoch+1) % SAVE_INTERVAL == 0:
            samples = generator(fixed_noise, training=False)
            save_image_grid(samples, epoch)

    # optional: save final models
    # generator.save("generator_final.h5")
    # discriminator.save("discriminator_final.h5")