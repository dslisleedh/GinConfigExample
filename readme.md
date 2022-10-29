# Simple MNIST Classification Project Using Gin Config and TF2

-------------

This is a simple MNIST classification project using [Gin Config](https://github.com/google/gin-config) and Tensorflow2

You can train MNIST Classifier by `train.py`

    conda env create -f environment.yaml
    conda activate gin
    python train.py

You can easily change model and log save path by `./conf/config.yaml`
    
    # ./conf/config.yaml
    model_name: "mlp"
    save_path: './logs'

Model parameters determined by `./conf/models/[model_name]_config.gin`
    
    # ./conf/models/mlpmixer_config.gin
    model_config.model = @MLPMixer()

    MLPMixer.config_intro = {
        'n_filters' : 128,
        'patch_size' : 4
    }
    MLPMixer.config_feature_extractor = {
        'n_layers': 8,
        'dropout_rate': .2,
        'act': @tf.nn.gelu,
        'expansion_rate': 4
    }
    MLPMixer.config_classifier = {
        'n_filters': (),
        'act': @tf.nn.relu,
        'dropout_rate': 0.5,
        'n_classes': 10
    }
    
    model_config.optimizer = @tf.keras.optimizers.Adam()
    tf.keras.optimizers.Adam.learning_rate = 1e-5
    model_config.loss_fn = @tf.keras.losses.SparseCategoricalCrossentropy()
    tf.keras.losses.SparseCategoricalCrossentropy.from_logits = True
    model_config.metrics = [@tf.keras.metrics.SparseCategoricalAccuracy()]
    
    model_config.batch_size = 256
    model_config.epochs = 100
    model_config.patience = 10



## Implemented models

- Classic MLP
- VGGNet
- ResNet
- MLP-Mixer
