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

Both paths provided are absolute paths, but they could also be relative paths. Absolute paths are provided to avoid any assumptions on the current working directory. Any value between *'s needs to provided. Note: *trial-name* is typically a long string, so we recommend renaming it to something more meaningful. The format of *checkpoint_number* is something like "checkpoint_00770" or "checkpoint_001000"; however, we also recommend saving the checkpoint name as something more meaningful depending on what metric you're observing.


## Tensorboard
You can track training or view the results of training by using tensorboard.
```
tensorboard --logdir=tensorboard_logs
```

## **Citation**

If you find the code useful for your work, please star this repo and consider citing:

```
@article{wang2023intersection,
  title={Learning to Control and Coordinate Mixed Traffic Through Robot Vehicles at Complex and Unsignalized Intersections},
  author={Wang, Dawei and Li, Weizi and Zhu, Lei and Pan, Jia},
  journal={arXiv preprint arXiv:2301.05294},
  year={2023}
}
```
