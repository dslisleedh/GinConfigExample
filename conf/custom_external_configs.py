import gin
import tensorflow as tf


gin.external_configurable(tf.nn.gelu, 'tf.nn.gelu')

gin.external_configurable(
    tf.keras.losses.SparseCategoricalCrossentropy, 'tf.keras.losses.SparseCategoricalCrossentropy')

gin.external_configurable(
    tf.keras.metrics.SparseCategoricalAccuracy, 'tf.keras.metrics.SparseCategoricalAccuracy')