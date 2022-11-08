import tensorflow as tf
import tensorflow_datasets as tfds
from src import (layers, models)

import gin.tf.external_configurables
import hydra
from hydra.utils import get_original_cwd

import os
import time
from omegaconf import OmegaConf
from functools import partial
from typing import List, Tuple, Union, Optional, Sequence
from utils import *


def train(
    model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: tf.keras.losses.Loss, metrics: List[tf.keras.metrics.Metric],
    epochs: int, batch_size: int, patience: int
):
    with open('./config.gin', 'w') as f:
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
            log_dir='./logs', histogram_freq=1, update_freq='batch'
        )
    ]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    model.fit(
        train_ds, epochs=epochs, validation_data=valid_ds, callbacks=callbacks
    )

    print('\nStart evaluating...')
    result = model.evaluate(test_ds)
    print('\nTrain Result:')
    with open('./result.txt', 'w') as f:
        for metric, value in zip(model.metrics_names, result):
            print(f'{metric}: {value}')
            f.write(f'{metric}: {value} \n')

    print('\nSaving model...')
    model.save_weights('./model_weights')


@hydra.main(config_path='./conf', config_name='config', version_base=None)
def main(main_config):
    # To prevent gin from load the config multiple times when use --multirun
    @RunnerDecorator
    def _main():
        load_externel_configure()
        config_files = [
            get_original_cwd() + '/conf/models/' + main_config.model + '.gin',
            get_original_cwd() + '/conf/optimizer/config.gin',
            get_original_cwd() + '/conf/others/config.gin',
        ]
        gin.parse_config_files_and_bindings(config_files, None)
        config = load_model_configure()
        train(**config)

    _main()

if __name__ == '__main__':
    main()
