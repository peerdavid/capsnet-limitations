# Limitation of capsule networks
Official implementation of the paper "Limitation of Capsule Networks"

# Setup
For this installation we assume that python3, pip3 and all nvidia drivers
(GPU support) are already installed. Then execute the following
to create a virtual environment and install all necessary packages:

1. Create virtual environment: ```python3 -m venv env```
2. Activate venv: ```source env/bin/activate```
3. Update your pip installation: ```pip3 install --upgrade pip```
4. Install all requirements. Use requirements-gpu if a gpu is available, requirements-cpu otherwise: ```pip3 install -r requirements.txt```


# Execute experiments
To run the sign experiments use ```train_sgn.py``` and for the depth 
experiment the ```train.py``` file. All parameters (depth, num capsules, 
dimensions etc.) can be changed using command arguments. For example to 
enable a bias term run ```train.py --use_bias=True```

To evaluate the results after the training tensorboard can be used.
Note also that we assume a multi GPU setup for the training and therefore we use the 
distr. API from TensorFlow 2.0. To run experiments on multiple nodes the run_depth_experiment.sh
file can be split (e.g. depth per node or RBA/EM per node etc.). 
We also watn to mention that the same source will also be uploaded 
to GitHub to make all experiments, the architecture and all hyperparameters available and
reproducable.