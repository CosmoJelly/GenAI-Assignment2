import tensorflow as tf
from tensorflow.keras import layers, Model

class Encoder(Model):
    def __init__(self, z_dim):
        super().__init__()
        self.conv = tf.keras.Sequential([
            layers.Conv2D(32, 4, 2, padding='same', activation='relu'),
            layers.Conv2D(64, 4, 2, padding='same', activation='relu'),
            layers.Conv2D(128, 4, 2, padding='same', activation='relu'),
            layers.Flatten()
        ])
        self.mu = layers.Dense(z_dim)
        self.logvar = layers.Dense(z_dim)

    def call(self, x):
        h = self.conv(x)
        return self.mu(h), self.logvar(h)

class Decoder(Model):
    def __init__(self, z_dim):
        super().__init__()
        self.fc = layers.Dense(8 * 8 * 128, activation='relu')
        self.reshape = layers.Reshape((8, 8, 128))
        self.deconv = tf.keras.Sequential([
            layers.Conv2DTranspose(64, 4, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, 4, 2, padding='same', activation='relu'),
            layers.Conv2DTranspose(1, 4, 2, padding='same', activation='sigmoid')
        ])

    def call(self, z):
        x = self.fc(z)
        x = self.reshape(x)
        return self.deconv(x)

class VAE(Model):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.z_dim = z_dim

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.shape)
        return mu + tf.exp(0.5 * logvar) * eps

    def call(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def compute_loss(self, x):
        recon, mu, logvar = self.call(x)
        recon_loss = tf.reduce_mean(tf.square(x - recon))
        kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
        total = recon_loss + 0.001 * kl_loss
        return total, recon_loss, kl_loss