import tensorflow as tf
from data_loader import load_dataset
from models import build_generator, build_discriminator
from losses import generator_loss, discriminator_loss
from utils import display_predictions

EPOCHS = 100

# Add memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Memory growth enabled for GPU")
    except RuntimeError as e:
        print(e)

# Load datasets
train_ds = load_dataset('train')
val_ds = load_dataset('val')

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers
gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

@tf.function
def train_step(input_image, target):
    with tf.GradientTape(persistent=True) as tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target
        )
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    return gen_total_loss, disc_loss

def fit(train_ds, val_ds, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for sketch, photo in train_ds:
            g_loss, d_loss = train_step(sketch, photo)
        print(f"Generator loss: {g_loss:.4f} | Discriminator loss: {d_loss:.4f}")
        display_predictions(generator, val_ds, num=2)

if __name__ == "__main__":
    fit(train_ds, val_ds, EPOCHS)