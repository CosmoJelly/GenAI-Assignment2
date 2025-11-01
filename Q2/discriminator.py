import tensorflow as tf
from tensorflow.keras import layers, Model

# -------------------------
# Custom spectral conv layer
# -------------------------
class ConvSN2D(layers.Layer):
    """
    Convolution layer with spectral normalization applied to the kernel
    using power iteration; performs conv with normalized kernel (no kernel assignment).
    """
    def __init__(self, filters, kernel_size, strides=1, padding='same', use_bias=True, power_iters=1, kernel_initializer='glorot_uniform', **kw):
        super().__init__(**kw)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.strides = (1, strides, strides, 1) if isinstance(strides, int) else (1, strides[0], strides[1], 1)
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.power_iters = power_iters
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        kh, kw = self.kernel_size
        shape = (kh, kw, in_channels, self.filters)
        self.kernel = self.add_weight(name='kernel', shape=shape, initializer=self.kernel_initializer, trainable=True, dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True, dtype=self.dtype)
        else:
            self.bias = None
        # u vector for power iteration (shape [1, out_channels])
        self.u = self.add_weight(name='sn_u', shape=(1, self.filters), initializer=tf.random_normal_initializer(), trainable=False, dtype=self.dtype)

    def compute_spectral_norm(self, w, u, iters=1):
        # w: kernel reshaped to [-1, out_channels]
        w_reshaped = tf.reshape(w, [-1, tf.shape(w)[-1]])  # [K, out]
        u_ = u
        for _ in range(iters):
            v_ = tf.math.l2_normalize(tf.matmul(u_, tf.transpose(w_reshaped)))
            u_ = tf.math.l2_normalize(tf.matmul(v_, w_reshaped))
        # sigma = u^T W v
        v_ = tf.stop_gradient(v_)
        u_ = tf.stop_gradient(u_)
        sigma = tf.matmul(tf.matmul(v_, w_reshaped), tf.transpose(u_))
        return sigma, u_

    def call(self, inputs, training=None):
        # compute spectral norm
        sigma, u_ = self.compute_spectral_norm(self.kernel, self.u, self.power_iters)
        w_bar = self.kernel / sigma
        # conv via tf.nn.conv2d using normalized kernel
        x = tf.nn.conv2d(inputs, w_bar, strides=self.strides, padding=self.padding)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        # update u vector
        self.u.assign(u_)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "power_iters": self.power_iters,
        })
        return cfg

# -------------------------
# Minibatch standard deviation
# -------------------------
class MinibatchStdDev(layers.Layer):
    def __init__(self, group_size=4, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size
        self.epsilon = epsilon

    def call(self, x):
        # x shape: [N, H, W, C]
        N, H, W, C = tf.unstack(tf.shape(x))
        group_size = tf.minimum(self.group_size, N)
        
        # Reshape into [G, M, H, W, C]
        y = tf.reshape(x, [group_size, -1, H, W, C])
        mean = tf.reduce_mean(y, axis=0, keepdims=True)
        var = tf.reduce_mean((y - mean) ** 2, axis=0)
        std = tf.sqrt(var + self.epsilon)

        # Mean over channels and spatial dimensions -> [M, 1, 1, 1]
        mean_std = tf.reduce_mean(std, axis=[1, 2, 3], keepdims=True)

        # Tile and concatenate
        mean_std = tf.tile(mean_std, [group_size, H, W, 1])  # [N, H, W, 1]
        return tf.concat([x, mean_std], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + 1,)

# -------------------------
# Feature extractor & Siamese discriminator
# -------------------------
def build_feature_extractor():
    inp = layers.Input((32,32,3))
    x = layers.GaussianNoise(0.05)(inp)

    x = ConvSN2D(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = ConvSN2D(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = ConvSN2D(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = MinibatchStdDev()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)

    return Model(inp, x, name='feature_extractor')

def build_siamese_discriminator():
    feat = build_feature_extractor()

    i1 = layers.Input((32,32,3))
    i2 = layers.Input((32,32,3))

    f1 = feat(i1)
    f2 = feat(i2)

    diff = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([f1, f2])
    x = layers.Dense(256, activation='relu')(diff)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    return Model([i1,i2], out, name='siamese_discriminator')