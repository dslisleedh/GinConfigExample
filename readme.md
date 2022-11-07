# Simple MNIST Classification Project Using Gin Config and TF2

-------------

This is a simple MNIST classification project using [Gin Config](https://github.com/google/gin-config) and Tensorflow2

    conda env create -f environment.yaml
    conda activate gin
    python train.py

You can easily change model by overiding model argument
    
    python train.py model=mlpmixer

Hyperparameters are determined by these configs.
 - ./conf/models/[model_name].gin   # Model selection and hyperparameters
 - ./conf/optimizer/config.gin      # Optimizer, Metrices and Loss selection and hyperparameters
 - ./conf/others/config.gin         # Other train-related hyperparameters. ex) batch_size, epochs, ...
    
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

## Implemented models

- Classic MLP
- VGGNet
- ResNet
- MLP-Mixer
