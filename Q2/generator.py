import tensorflow as tf
from tensorflow.keras import layers, Model

def ResidualBlock(filters):
    inp = layers.Input((None, None, filters))
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    out = layers.Add()([inp, x])
    return Model(inp, out, name=f"res_{filters}")

def build_generator(latent_dim=100):
    """
    Generator that outputs 32x32x3 images in range [-1,1].
    """
    z = layers.Input(shape=(latent_dim,))

    x = layers.Dense(8 * 8 * 512, use_bias=False)(z)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((8, 8, 512))(x)

    # upsample to 16x16
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = ResidualBlock(256)(x)
    x = layers.Dropout(0.3)(x)

    # upsample to 32x32
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # refine
    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.GaussianNoise(0.03)(x)
    out = layers.Conv2D(3, 3, padding='same', activation='tanh')(x)

    return Model(z, out, name="Generator")