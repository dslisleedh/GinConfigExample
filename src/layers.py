import tensorflow as tf
import gin
from typing import List, Tuple, Union, Optional, Sequence


class MLP(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: Sequence[int], act: tf.nn, dropout_rate: float = 0.0
    ):
        super(MLP, self).__init__()

        self.forward = tf.keras.Sequential([])
        for filter in n_filters:
            self.forward.add(tf.keras.layers.Dense(filter, activation=act))
            self.forward.add(tf.keras.layers.Dropout(dropout_rate))

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


class VGGBlock(tf.keras.layers.Layer):
    def __init__(
            self,  n_filters: int, n_layers: int, act: tf.nn
    ):
        super(VGGBlock, self).__init__()

        self.forward = tf.keras.Sequential([
            *[
                tf.keras.layers.Conv2D(
                    n_filters, 3, padding='same', activation=act
                )
            ] * n_layers,
            tf.keras.layers.MaxPool2D(2, 2)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


# Implement Only Bottleneck-like block for convenience
class ResBlock(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: Sequence[int], act: tf.nn, increase_dim: bool = False,
            down_sample: bool = False
    ):
        super(ResBlock, self).__init__()

        # Use Lambda layer to activation for consistency
        self.forward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                n_filters[0], 1, strides=2 if down_sample else 1, padding='same'
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Lambda(act),
            tf.keras.layers.Conv2D(
                n_filters[1], 3, padding='same', activation=act
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Lambda(act),
            tf.keras.layers.Conv2D(
                n_filters[2], 1, padding='same', activation=act
            ),
            tf.keras.layers.BatchNormalization()
        ])

        if increase_dim:
            self.skip = tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    n_filters[2], 3 if down_sample else 1, strides=2 if down_sample else 1, padding='same'
                )
            ])
        else:
            self.skip = tf.keras.layers.Layer()

    def call(self, inputs, training=False, *args, **kwargs):
        return self.forward(inputs, training=training) + self.skip(inputs)


class MixerBlock(tf.keras.layers.Layer):
    def __init__(self, act: tf.nn, dropout_rate: float, expansion_rate: int = 4):
        super(MixerBlock, self).__init__()
        self.act = act
        self.dropout_rate = dropout_rate
        self.expansion_rate = expansion_rate

    def build(self, input_shape):
        # Input_shape: [batch, seq_len(h * w), dim]
        self.spatial = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(input_shape[-2] * self.expansion_rate),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Lambda(self.act),
            tf.keras.layers.Dense(input_shape[-2]),
            tf.keras.layers.Dropout(self.dropout_rate)
        ])

        self.channel = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(input_shape[-1] * self.expansion_rate),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Lambda(self.act),
            tf.keras.layers.Dense(input_shape[-1]),
            tf.keras.layers.Dropout(self.dropout_rate)
        ])

    def call(self, inputs, *args, **kwargs):
        inputs = tf.transpose(inputs, [0, 2, 1])
        inputs = inputs + self.spatial(inputs)
        inputs = tf.transpose(inputs, [0, 2, 1])
        inputs = inputs + self.channel(inputs)
        return inputs


class ClassificationHead(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: Sequence[int],  n_classes: int, act: Optional[tf.nn] = None,
            dropout_rate: float = 0.0
    ):
        super(ClassificationHead, self).__init__()

        self.forward = tf.keras.Sequential([])
        for n_filter in n_filters:
            self.forward.add(tf.keras.layers.Dense(n_filter, activation=act))
            self.forward.add(tf.keras.layers.Dropout(dropout_rate))
        self.forward.add(tf.keras.layers.Dense(n_classes))

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


# class PrintLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super(PrintLayer, self).__init__()
#
#     def call(self, inputs, *args, **kwargs):
#         print(inputs.shape)
#         return inputs
