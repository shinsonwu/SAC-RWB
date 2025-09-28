### Requirements
    Ubuntu

We highly recommend installing on Ubuntu 22.04 as this version of Ubuntu has been tested. If you have a Windows machine, we recommend using WSL to create an Ubuntu 22.04 virtual machine for setting up the repo.

### Install SUMO
Please follow the official instructions provided [here](https://sumo.dlr.de/docs/Installing/index.html).
It should involve just running the following command:
```
sudo apt-get install sumo sumo-tools sumo-doc
```
We recommend SUMO v1.12.0. If you use a later version of SUMO, be aware that the units for fuel consumption are different on later versions. The resulting number is the same as older versions, only the units are different.

### Setup Conda environment
    conda create --name SAC_RWB python=3.7
    conda activate SAC_RWB
    pip install -r requirements.txt

## Training
    
To train a model, an example command is the following:
```
python train_sac.py
```       

For reference, an example reward curve after training using the above command should look like:

## Evaluation
To evaluate a trained model, an example command is the following:
```
python test_sac.py
```


## Tensorboard
You can track training or view the results of training by using tensorboard.
```
tensorboard --logdir=tensorboard_logs
```
