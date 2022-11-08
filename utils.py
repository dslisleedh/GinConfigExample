import gin
import tensorflow as tf
from typing import List, Tuple, Union, Optional, Sequence


def load_externel_configure():
    gin.external_configurable(tf.nn.gelu, 'tf.nn.gelu')

    gin.external_configurable(
        tf.keras.losses.SparseCategoricalCrossentropy, 'tf.keras.losses.SparseCategoricalCrossentropy')

    gin.external_configurable(
        tf.keras.metrics.SparseCategoricalAccuracy, 'tf.keras.metrics.SparseCategoricalAccuracy')


@gin.configurable(name_or_fn='model_config')
def load_model_configure(
        model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: tf.keras.losses.Loss, metrics: List[tf.keras.metrics.Metric],
        epochs: int, batch_size: int, patience: int
):
    cfg = {
        'model': model,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'metrics': metrics,
        'epochs': epochs,
        'batch_size': batch_size,
        'patience': patience
    }
    return cfg


def preprocessing(x, y, augment: bool = False):
    x = tf.cast(x, tf.float32) / 255.
    if augment:
        b, _, _, _ = x.get_shape().as_list()
        x = tf.image.random_crop(x, [b, 26, 26, 1])
        x = tf.image.resize(x, [28, 28])
    return x, y


class RunnerDecorator:
    def __init__(self, f):
        self.f = f

    def __call__(self):
        try :
            self.f()
            gin.clear_config()
        except Exception as e:
            print(e)
            gin.clear_config()
            raise e
