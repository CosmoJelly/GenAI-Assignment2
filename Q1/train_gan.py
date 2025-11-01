import tensorflow as tf
import os
from models.gan import Generator, Discriminator
from utils.dataset import get_signature_dataset
from utils.visualize import show_images
from utils.metrics import compute_ssim
import config

train_ds, _ = get_signature_dataset(config.DATA_DIR, config.IMG_SIZE, config.BATCH_SIZE)
generator = Generator(config.Z_DIM)
discriminator = Discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
gen_opt = tf.keras.optimizers.Adam(config.GAN_LR, beta_1=0.5)
disc_opt = tf.keras.optimizers.Adam(config.GAN_LR, beta_1=0.5)

@tf.function
def train_step(images):
    noise = tf.random.normal([config.BATCH_SIZE, config.Z_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(noise)
        real_output = discriminator(images)
        fake_output = discriminator(generated)
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = (cross_entropy(tf.ones_like(real_output), real_output) +
                     cross_entropy(tf.zeros_like(fake_output), fake_output))
    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gen_opt.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))
    return gen_loss, disc_loss

for epoch in range(config.EPOCHS_GAN):
    for batch in train_ds:
        g_loss, d_loss = train_step(batch)
    print(f"Epoch {epoch+1}/{config.EPOCHS_GAN}: G={g_loss:.4f}, D={d_loss:.4f}")
    noise = tf.random.normal([8, config.Z_DIM])
    samples = generator(noise)

real_batch = next(iter(train_ds))
real_batch = real_batch[:8]
ssim_score = compute_ssim(real_batch, samples)
print(f"GAN Test SSIM: {ssim_score}")
show_images(samples)
