import tensorflow as tf
import tensorflow_datasets as tfds
from src import (layers, models)

import gin.tf.external_configurables
import conf.custom_external_configs

import os
import time
from functools import partial
from typing import List, Tuple, Union, Optional, Sequence


def preprocessing(x, y, augment: bool = False):
    x = tf.cast(x, tf.float32) / 255.
    if augment:
        b, _, _, _ = x.get_shape().as_list()
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_crop(x, [b, 26, 26, 1])
        x = tf.image.resize(x, [28, 28])
    return x, y


@gin.configurable()
def train(
    model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss, metrics: List[tf.keras.metrics.Metric],
    epochs: int, batch_size: int, patience: int,
    save_path: str
):
    time_now = time.localtime(time.time())
    save_path = save_path + f'/{model.name}/' + time.strftime('%Y%m%d%H%M%S', time_now)
    os.makedirs(save_path, exist_ok=False)
    with open(save_path + '/simplemlp_config.gin', 'w') as f:
        f.write(gin.operative_config_str())

    print('\nLoding dataset...')
    train_ds, valid_ds, test_ds = tfds.load(
        'mnist', as_supervised=True, split=['train[:80%]', 'train[80%:]', 'test']
    )
    aug_preprocessing = partial(preprocessing, augment=True)
    train_ds = train_ds.shuffle(10000).batch(batch_size, drop_remainder=True).\
        map(aug_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(batch_size, drop_remainder=False).\
        map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE).\
        prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=False).\
        map(preprocessing, num_parallel_calls=tf.data.AUTOTUNE).\
        prefetch(tf.data.AUTOTUNE)

    print('Start training...')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=save_path + '/logs', histogram_freq=1, update_freq='batch'
        )
    ]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    model.fit(
        train_ds, epochs=epochs, validation_data=valid_ds, callbacks=callbacks
    )

    print('\nStart evaluating...')
    result = model.evaluate(test_ds)
    print('\nTrain Result:')
    with open(save_path + '/result.txt', 'w') as f:
        for metric, value in zip(model.metrics_names, result):
            print(f'{metric}: {value}')
            f.write(f'{metric}: {value} \n')

    print('\nSaving model...')
    model.save_weights(save_path + '/model_weights')


if __name__ == '__main__':
    gin.parse_config_file('./conf/resnet_config.gin')
    train()