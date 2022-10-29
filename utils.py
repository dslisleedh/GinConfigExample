import gin
import tensorflow as tf


def load_externel_configure():
    gin.external_configurable(tf.nn.gelu, 'tf.nn.gelu')

    gin.external_configurable(
        tf.keras.losses.SparseCategoricalCrossentropy, 'tf.keras.losses.SparseCategoricalCrossentropy')

    gin.external_configurable(
        tf.keras.metrics.SparseCategoricalAccuracy, 'tf.keras.metrics.SparseCategoricalAccuracy')


def preprocessing(x, y, augment: bool = False):
    x = tf.cast(x, tf.float32) / 255.
    if augment:
        b, _, _, _ = x.get_shape().as_list()
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_crop(x, [b, 26, 26, 1])
        x = tf.image.resize(x, [28, 28])
    return x, y
