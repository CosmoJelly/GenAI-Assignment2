import tensorflow as tf
from tensorflow.keras import layers, Model

class Generator(Model):
    def __init__(self, z_dim):
        super().__init__()
        self.model = tf.keras.Sequential([
            layers.Dense(8 * 8 * 256, use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Reshape((8, 8, 256)),
            layers.Conv2DTranspose(128, 4, 2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(64, 4, 2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(1, 4, 2, padding='same', activation='tanh')
        ])

    def call(self, z):
        return self.model(z)

class Discriminator(Model):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, 4, 2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, 4, 2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        return self.model(x)