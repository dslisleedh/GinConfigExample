import gin
import tensorflow as tf
from src.layers import *
from einops.layers.keras import Rearrange


@gin.configurable
class SimpleMLP(tf.keras.models.Model):
    def __init__(
            self, config_feature_extractor: dict, config_classifier: dict,
            name: str = 'SimpleMLP'
    ):
        super(SimpleMLP, self).__init__(name=name)

        self.forward = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            MLP(**config_feature_extractor),
            ClassificationHead(**config_classifier)
        ])

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)

@gin.configurable
class VGGNet(tf.keras.models.Model):
    def __init__(
            self, config_feature_extractor: dict, config_classifier: dict,
            name: str = 'VGGNet'
    ):
        super(VGGNet, self).__init__(name=name)

        self.forward = tf.keras.Sequential([])
        for n_filters, n_layers in zip(config_feature_extractor['n_filters'], config_feature_extractor['n_layers']):
            self.forward.add(VGGBlock(n_filters, n_layers, config_feature_extractor['act']))
        self.forward.add(tf.keras.layers.Flatten())
        self.forward.add(ClassificationHead(**config_classifier))

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)

@gin.configurable
class ResNet(tf.keras.models.Model):
    def __init__(
            self, config_intro: dict, config_feature_extractor: dict, config_classifier: dict,
            name: str = 'ResNet'
    ):
        super(ResNet, self).__init__(name=name)

        self.forward = tf.keras.Sequential([])
        # Intro
        self.forward.add(tf.keras.layers.Conv2D(
            config_intro['n_filters'], 3, padding='same'
        ))
        self.forward.add(tf.keras.layers.BatchNormalization())
        self.forward.add(tf.keras.layers.Lambda(config_feature_extractor['act']))
        # Feature Extractor
        for n_filters, n_layers, increase_dim, down_sample in zip(
                config_feature_extractor['n_filters'],
                config_feature_extractor['n_layers'],
                config_feature_extractor['increase_dim'],
                config_feature_extractor['down_sample']
        ):
            for _ in range(n_layers):
                self.forward.add(
                    ResBlock(n_filters, config_feature_extractor['act'], increase_dim, down_sample)
                )
                increase_dim = False
                down_sample = False
        self.forward.add(tf.keras.layers.GlobalAveragePooling2D())
        # Head
        self.forward.add(ClassificationHead(**config_classifier))

    def call(self, inputs, *args, **kwargs):
        return self.forward(inputs)


@gin.configurable
class MLPMixer(tf.keras.models.Model):
    def __init__(
            self, config_intro: dict, config_feature_extractor: dict, config_classifier: dict,
    ):
        super(MLPMixer, self).__init__()

        self.forward = tf.keras.Sequential([])
        # Intro
        self.forward.add(
            tf.keras.layers.Conv2D(
                config_intro['n_filters'], config_intro['patch_size'], strides=config_intro['patch_size'],
                padding='same'
            )
        )
        self.forward.add(
            Rearrange('b h w c -> b (h w) c')
        )
        # Feature Extractor
        n_layers = config_feature_extractor.pop('n_layers')
        for _ in range(n_layers):
            self.forward.add(
                MixerBlock(**config_feature_extractor)
            )
        self.forward.add(tf.keras.layers.GlobalAveragePooling1D())
        # Head
        self.forward.add(ClassificationHead(**config_classifier))

    def call(self, inputs, training: bool = False, *args, **kwargs):
        return self.forward(inputs, training=training)
